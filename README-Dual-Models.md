---

# 🚀 Dual-Model Deployment: Qwen3.6-27B & GPT-OSS-20B on NVIDIA DGX Spark (ARM64)

คู่มือนี้สาธิตวิธีการรันโมเดล Large Language Models (LLMs) ประสิทธิภาพสูง **2 ตัวพร้อมกัน (Concurrent Execution)** บนเซิร์ฟเวอร์สถาปัตยกรรม ARM64 (เช่น NVIDIA DGX Spark, Grace-Blackwell GB10) ผ่าน `llama.cpp` และ Docker

การตั้งค่านี้ออกแบบมาเพื่อระบบ Multi-Agent AI ที่ต้องการเรียกใช้โมเดลเฉพาะทางหลายตัวพร้อมกัน โดยใช้ประโยชน์จาก VRAM มหาศาลของการ์ดจอระดับองค์กรอย่างเต็มประสิทธิภาพ

## 🧠 การวิเคราะห์ทรัพยากร (Resource Allocation)

* **Model 1: Qwen3.6-27B (Q8_K_XL)** ใช้ VRAM ประมาณ ~34 GB
* **Model 2: GPT-OSS-20B (Q8_K_XL)** ใช้ VRAM ประมาณ ~21 GB
* **รวมการใช้ VRAM:** ~55 GB (สามารถรันบน NVIDIA GB10 GPU ตัวเดียวกันได้อย่างสบายโดยไม่เกิดคอขวด)
* **การจัดการ Network:** รันแยก Container อย่างสมบูรณ์ โมเดลแรกใช้พอร์ต `8080` โมเดลที่สองใช้พอร์ต `8081`

---

## 🛠️ ขั้นตอนการติดตั้ง (Deployment Guide)

### 1. ดาวน์โหลดไฟล์โมเดล (Manual Download)

เพื่อป้องกันปัญหา Network Timeout จากการให้ Docker โหลดไฟล์ขนาดใหญ่ เราจะดาวน์โหลดโมเดลทั้งสองตัวมาไว้ที่เครื่องโฮสต์ก่อน

```bash
mkdir -p qwen-agent-server/models
cd qwen-agent-server

# โหลด Qwen3.6-27B (~34GB)
wget -c https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-Q8_K_XL.gguf -P models/

# โหลด GPT-OSS-20B (~21GB)
wget -c https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-UD-Q8_K_XL.gguf -P models/

```

### 2. สร้างไฟล์ `Dockerfile` (ARM64 CUDA Stubs Workaround)

เนื่องจากเซิร์ฟเวอร์เป็นสถาปัตยกรรม ARM64 การคอมไพล์ CUDA ภายใน Docker มักจะหาไดรเวอร์ GPU ไม่เจอ (Missing `libcuda.so.1`) เราจะใช้เทคนิคสร้าง Symlink ไปยังไฟล์จำลองชั่วคราวเพื่อผ่านขั้นตอนนี้

สร้างไฟล์ `Dockerfile` ภายในโฟลเดอร์โปรเจกต์:

```dockerfile
# ใช้ Base Image ที่รองรับ CUDA และ ARM64 จาก NVIDIA
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ติดตั้งเครื่องมือ
RUN apt-get update && apt-get install -y git build-essential cmake wget

# โหลดซอร์สโค้ด llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git /app
WORKDIR /app

# สร้าง Symlink ไปยังไฟล์ไดรเวอร์จำลอง (CUDA Stubs) สำหรับตอน Build
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 || true

# ตั้งค่า CMake ให้เปิดใช้การ์ดจอ (GGML_CUDA=ON)
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# สั่ง Compile โดยหลอก Path ให้ไปอ่านไฟล์จำลองชั่วคราว
RUN LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}" \
    cmake --build build --config Release -j $(nproc)

EXPOSE 8080
ENTRYPOINT ["/app/build/bin/llama-server"]

```

### 3. สร้างไฟล์ `docker-compose.yml` (Dual Services)

ตั้งค่าให้ Docker รัน Container 2 ตัวจาก Dockerfile เดียวกัน แต่เรียกใช้ไฟล์โมเดลและพอร์ตที่ต่างกัน

```yaml
version: '3.8'

services:
  # ----------------------------------------------------
  # Model 1 : Qwen3.6-27B (Port: 8080)
  # ----------------------------------------------------
  qwen-server:
    build: .
    container_name: qwen3.6-agent-server
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    command: >
      -m /models/Qwen3.6-27B-UD-Q8_K_XL.gguf
      -ngl 999
      -fa on
      -c 32768
      --reasoning-preserve
      --host 0.0.0.0
      --port 8080
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  # ----------------------------------------------------
  # Model 2 : GPT-OSS-20B (Port: 8081)
  # ----------------------------------------------------
  gpt-oss-server:
    build: .
    container_name: gpt-oss-20b-server
    restart: unless-stopped
    ports:
      - "8081:8080"  # Map พอร์ตภายนอก 8081 ไปยังพอร์ต 8080 ภายในคอนเทนเนอร์
    volumes:
      - ./models:/models
    command: >
      -m /models/gpt-oss-20b-UD-Q8_K_XL.gguf
      -ngl 999
      -fa on
      -c 32768
      --host 0.0.0.0
      --port 8080
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

```

### 4. รันระบบและตรวจสอบ (Deploy & Verify)

สั่งรันระบบ (การ Build จะเกิดขึ้นเพียงครั้งเดียว และใช้ Image ร่วมกันทั้ง 2 คอนเทนเนอร์)

```bash
docker compose up -d --build

```

ตรวจสอบ Log การทำงานของแต่ละโมเดลว่าสามารถ Offload ขึ้น GPU ได้สำเร็จ:

```bash
# เช็ค Log ของโมเดล Qwen (ควรเห็นพอร์ต 8080)
docker logs -f qwen3.6-agent-server

# เช็ค Log ของโมเดล GPT-OSS (ควรเห็นการทำงานคู่ขนานกัน)
docker logs -f gpt-oss-20b-server

```

> **📌 Note สำหรับ Log ของ GPT-OSS-20B:**
> หากพบข้อความแจ้งเตือนสีเหลือง (Warning) เช่น `setting token '<|message|>' attribute to USER_DEFINED` ถือว่าเป็น **เรื่องปกติ** เนื่องจากผู้พัฒนา (Unsloth) ได้เพิ่ม Special Tokens สำหรับงานด้าน Agent เอาไว้ โมเดลยังคงทำงานได้ 100%

---

## 📡 ทดสอบการเรียกใช้งาน API (API Testing)

คุณสามารถเชื่อมต่อกับโมเดลทั้งสองได้ทันทีผ่าน OpenAI-Compatible API:

**ทดสอบ Qwen3.6 (พอร์ต 8080):**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "You are Qwen. Introduce yourself briefly."}]}'

```

**ทดสอบ GPT-OSS (พอร์ต 8081):**

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "You are GPT-OSS. Introduce yourself briefly."}]}'

```
