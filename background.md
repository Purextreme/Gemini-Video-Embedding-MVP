# Gemini Video Embedding 2 视频检索验证项目 (MVP)

> **验证结论：Gemini-embedding-2-preview 对单镜头视频的语义理解极其出色，测试命中率 100%。**

本项目是一个最小可行性验证（MVP），旨在测试 Google 最新发布的 `gemini-embedding-2-preview` 模型在“视频-文本”跨模态检索中的实际效果。

---

## 🚀 核心实验结果

通过对 22 个单镜头素材的测试，我们得到了以下关键数据：
- **模型**: `gemini-embedding-2-preview`
- **向量维度**: 3072 dim
- **Top-1 命中率**: **100%** (6/6 随机复杂指令测试)
- **平均相似度得分**: 0.35 - 0.49

---

## 🛠️ 技术实现流程

1.  **视频预处理**: 使用 `ffmpeg` 将原始 4K/1080p 素材转换为 720p 低码率 MP4，以优化上传带宽。
2.  **向量生成 (Inline Mode)**: 绕过存储 API 限制，直接通过 Base64/字节流将视频推入模型生成 3072 维向量。
3.  **语义检索**: 将自然语言查询（Query）同样转化为向量，计算 **余弦相似度** 并输出结果。

---

## 🖼️ 素材库预览 (Thumbnails)

以下是本项目测试所使用的部分单镜头素材首帧：

| | | | |
| :---: | :---: | :---: | :---: |
| ![01](thumbnails/01.jpg) | ![02](thumbnails/02.jpg) | ![03_1](thumbnails/03_1.jpg) | ![07_1](thumbnails/07_1.jpg) |
| ![09](thumbnails/09.jpg) | ![10](thumbnails/10.jpg) | ![11](thumbnails/11.jpg) | ![14](thumbnails/14.jpg) |
| ![15](thumbnails/15.jpg) | ![17](thumbnails/17.jpg) | ![22](thumbnails/22.jpg) | ![25](thumbnails/25.jpg) |

---

## 🔍 检索案例展示

| 检索指令 (Query) | 匹配视频 (Top-1) | 相似度 | 画面内容 |
| :--- | :--- | :--- | :---: |
| 一个小女孩在公园面吹泡泡 | `IMG_1284.mp4` | 0.4422 | ![IMG_1284](thumbnails/IMG_1284.jpg) |
| 一个拿着笔记本电脑的女性在办公室内行走 | `03_1.mp4` | 0.3994 | ![03_1](thumbnails/03_1.jpg) |
| 一只手把硬盘推入桌面上的NAS | `25.mp4` | 0.4889 | ![25](thumbnails/25.jpg) |
| 公园的板凳上放着无人机、遥控器 | `14.mp4` | 0.4786 | ![14](thumbnails/14.jpg) |

---

## 📂 项目结构说明

- `generate_embeddings.py`: 核心脚本，负责视频上传与向量化。
- `search_video.py`: 检索脚本，执行语义搜索逻辑。
- `convert_videos.py`: 视频批量转码工具。
- `embeddings.json`: 已持久化的视频特征数据库。
- `测试结果.md`: 详细的实验数据报告。

---

## 🧑‍💻 作者
**牛奶瓶**

---
> 这是一个基于 Gemini CLI 自动生成的验证项目。
