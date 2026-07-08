
---

# 🚀 Deploy Qwen3.6-27B on NVIDIA DGX Spark (ARM64) via Docker & llama.cpp

คู่มือและไฟล์การตั้งค่า (Configuration Files) สำหรับการรันโมเดล **Qwen3.6-27B (UD-Q8_K_XL)** บนเครื่องเซิร์ฟเวอร์สถาปัตยกรรม **ARM64 + NVIDIA GPU** (เช่น NVIDIA DGX Spark, Grace-Blackwell GB10) ผ่านระบบ Docker โดยใช้ `llama.cpp`

โปรเจกต์นี้ออกแบบมาเพื่อแก้ปัญหาการคอมไพล์ CUDA ภายใน Docker บนเครื่องตระกูล ARM64 ซึ่งมักจะพบปัญหาหาไดรเวอร์ GPU ไม่เจอ (CUDA Stubs Issue) เพื่อให้สามารถดึงประสิทธิภาพของการ์ดจอมาใช้ในงาน **Agent AI** และ **Code Generation** ได้อย่างเต็ม 100%

## ✨ ฟีเจอร์หลัก (Features)

* 🏗️ **Native ARM64 Build:** คอมไพล์ `llama.cpp` จากซอร์สโค้ดเพื่อรองรับชิป Grace CPU (aarch64) โดยเฉพาะ
* ⚡ **Full GPU Offloading:** ยกโมเดลเข้า VRAM ทั้งหมด (NVIDIA Blackwell GB10 / CUDA 13.0)
* 🛠️ **CUDA Stubs Workaround:** ข้ามขีดจำกัดการคอมไพล์ GPU Driver ภายใน Docker
* 🧠 **Reasoning Mode Supported:** ตั้งค่า Context Window กว้างถึง 32K พร้อมเปิดโหมด `reasoning-preserve` สำหรับงาน Agent AI

---

## 📋 สิ่งที่ต้องมี (Prerequisites)

* ระบบปฏิบัติการ Ubuntu 22.04+ (สถาปัตยกรรม `arm64`)
* การ์ดจอ NVIDIA ที่รองรับ CUDA
* ติดตั้ง Docker และ Docker Compose เรียบร้อยแล้ว

---

## ⚙️ ขั้นตอนการติดตั้ง (Installation Guide)

### 1. ติดตั้ง NVIDIA Container Toolkit

จำเป็นต้องตั้งค่าให้ Docker สามารถสื่อสารกับ GPU ของเครื่องโฮสต์ได้

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

```

### 2. เตรียมโฟลเดอร์และดาวน์โหลดโมเดล

เราจะทำการดาวน์โหลดโมเดลแบบ Manual เพื่อป้องกันปัญหา Network Timeout จากการโหลดผ่าน Container

```bash
# สร้างโฟลเดอร์สำหรับโปรเจกต์
mkdir -p qwen-agent-server/models
cd qwen-agent-server

# ดาวน์โหลดโมเดล Qwen3.6-27B (Q8_K_XL) ขนาด ~34GB
wget -c https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-Q8_K_XL.gguf -P models/

```

### 3. ตั้งค่า Dockerfile

สร้างไฟล์ `Dockerfile` ภายในโฟลเดอร์ `qwen-agent-server` โค้ดส่วนนี้คือ "หัวใจสำคัญ" ที่ช่วยให้คอมไพล์ผ่านบน ARM64 โดยใช้เทคนิคชี้ Path ไปหาไฟล์จำลอง (Stubs) ชั่วคราว

```dockerfile
# ใช้ Base Image ที่รองรับ CUDA และ ARM64
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ติดตั้งเครื่องมือพื้นฐาน
RUN apt-get update && apt-get install -y git build-essential cmake wget

# โหลดซอร์สโค้ด llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git /app
WORKDIR /app

# สร้าง Symlink สำหรับไฟล์จำลองไดรเวอร์ (CUDA Stubs) เพื่อใช้ตอน Build
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 || true

# สร้าง Build directory และเปิดใช้งาน CUDA
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# สั่ง Compile โดยชี้ตัวแปรสภาพแวดล้อมไปหาไฟล์จำลองชั่วคราว
RUN LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}" \
    cmake --build build --config Release -j $(nproc)

EXPOSE 8080
ENTRYPOINT ["/app/build/bin/llama-server"]

```

### 4. ตั้งค่า Docker Compose

สร้างไฟล์ `docker-compose.yml` เพื่อจัดการพอร์ต, การเชื่อมต่อ GPU และพารามิเตอร์ของโมเดล

```yaml
services:
  llama-server:
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

```

---

## 🚀 สั่งรันเซิร์ฟเวอร์ (Deploy)

เมื่อเตรียมไฟล์ทั้งหมดครบแล้ว ให้ใช้คำสั่งต่อไปนี้เพื่อคอมไพล์และสตาร์ทเซิร์ฟเวอร์ (การคอมไพล์ครั้งแรกอาจใช้เวลา 3-5 นาที)

```bash
docker compose up -d --build

```

ตรวจสอบสถานะการทำงาน (หากสำเร็จจะขึ้นข้อความ `HTTP server listening on 0.0.0.0:8080`):

```bash
docker compose logs -f

```

---

## 📡 การทดสอบใช้งาน (Usage Example)

ระบบจะเปิด API มาตรฐานที่เข้ากันได้กับ OpenAI (OpenAI-Compatible API) บนพอร์ต `8080` สามารถทดสอบการทำงานของ Agent ด้วย cURL:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "system", "content": "You are an expert AI programming assistant." },
      { "role": "user", "content": "Write a Python script to scan open ports on localhost." }
    ],
    "temperature": 0.2
  }'

```

**การตั้งค่าสำหรับ Tool/Agent (เช่น LangChain, Cursor, AutoGen):**

* **Base URL:** `http://<IP_เครื่องของคุณ>:8080/v1`
* **API Key:** `sk-1234` (ใส่ค่าอะไรก็ได้)
* **Model Name:** `qwen3.6-27b` (หรือชื่อใดก็ได้ตาม Tool ต้องการ)
