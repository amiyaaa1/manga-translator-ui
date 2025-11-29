FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（给 PyTorch / OpenCV / Pillow 等用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# 把项目代码拷进容器
COPY . /app

# 安装 CPU 版本依赖
RUN pip install --no-cache-dir -r requirements_cpu.txt

# 暴露 Web API 端口
EXPOSE 8000

# 默认启动 Web 模式服务，自动读取平台提供的 PORT 环境变量
CMD ["sh", "-c", "python -m manga_translator web --host 0.0.0.0 --port ${PORT:-8000}"]
