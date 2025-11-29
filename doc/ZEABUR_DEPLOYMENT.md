# 在 Zeabur 部署 Manga Translator UI（小白友好版）

本文档介绍如何在 Zeabur 上用官方 Dockerfile 部署 Web API 服务（CPU 版本）。

> 提示：Zeabur 使用容器运行，本项目的 `Dockerfile` 默认启动 Web API（`python -m manga_translator web --host 0.0.0.0 --port 8000`）。

## 准备工作

1. 注册并登录 [Zeabur](https://zeabur.com/) 账号。
2. 在 GitHub 中 fork 或直接使用本仓库，确保仓库可被 Zeabur 访问。
3. （可选）准备好翻译所需的 API Key，并在 Zeabur 中添加为环境变量。

## 创建服务

1. 打开 Zeabur 控制台，点击 **Create Service**。
2. 选择 **Deploy from Git**。
3. 绑定你的 GitHub 账号，选择包含本项目的仓库和分支（默认 `main`）。
4. 部署方式选择 **Dockerfile**，Zeabur 会自动识别根目录的 `Dockerfile`。
5. 保持默认构建命令（使用仓库内的 Dockerfile）。
6. 确认暴露端口为 **8000**（Dockerfile 已 EXPOSE 8000）。

## 配置环境变量

在服务的 **Environment Variables** 页添加常用配置：

- `MT_HOST`：服务绑定地址，保持 `0.0.0.0`
- `MT_PORT`：服务端口，保持 `8000`
- （可选）翻译器的 API Key，例如 `OPENAI_API_KEY`、`GEMINI_API_KEY`、`DEEPL_API_KEY` 等，根据你在应用中选择的翻译器填写。

> Dockerfile 中的默认启动命令已经设置为 `python -m manga_translator web --host 0.0.0.0 --port 8000`，因此不需要额外的启动命令。

## 发布与访问

1. 点击 **Deploy** 开始构建。首次构建会下载依赖，耗时取决于网络。
2. 构建成功后，Zeabur 会显示访问地址，例如 `https://your-service.zeabur.app`。
3. 打开浏览器访问该地址，即可使用 Web API。可在命令行通过 `curl` 访问：

```bash
curl https://your-service.zeabur.app/docs
```

4. 如果想更新代码，直接推送到同一分支，Zeabur 会自动触发重新部署。

## 常见问题

- **提示依赖安装失败**：稍后重试，或在仓库设置中开启 Zeabur 的缓存。也可在 `scripts/setup_linux.sh` 本地安装并锁定依赖版本后再推送。
- **需要 GPU 加速吗？**：Zeabur 的标准容器为 CPU 环境，Dockerfile 也默认安装 CPU 依赖。需要 GPU 时请使用本地或其他支持 GPU 的平台。
- **端口无法访问**：确认服务暴露端口为 8000，并确保未修改启动命令。
