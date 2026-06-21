<div align="center">

# 🌌 Intelligent Malayalam Document Understanding Framework

<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=32&duration=3000&pause=1000&color=00F7FF&center=true&vCenter=true&width=1000&lines=Multimodal+Document+Intelligence+Framework;YOLO+Powered+Layout+Analysis;Gemini+Vision+Semantic+Reasoning;Hierarchical+Multi-Label+Classification" />

<br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00c6ff,100:0072ff&height=200&section=header&text=KSHB%20Project&fontSize=50&fontColor=ffffff&animation=fadeIn"/>

<br>

![](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![](https://img.shields.io/badge/YOLO-Ultralytics-red?style=for-the-badge)
![](https://img.shields.io/badge/Gemini-2.5_Flash-purple?style=for-the-badge)
![](https://img.shields.io/badge/Computer-Vision-success?style=for-the-badge)
![](https://img.shields.io/badge/Document-AI-green?style=for-the-badge)
![](https://img.shields.io/badge/Multimodal-AI-orange?style=for-the-badge)

</div>

---

# ✨ Overview

> A multimodal document intelligence framework integrating spatial layout parsing and semantic reasoning for automated understanding and hierarchical organization of Malayalam official correspondence.

---

# ⚡ Pipeline

```mermaid
flowchart TD

A[📄 Input Document]
-->B[🎯 YOLO Layout Analysis]

B
-->C[✂ Region Extraction]

C
-->D[🧠 Gemini Vision]

D
-->E[🔍 Entity Recognition]

E
-->F[🏷 Multi-label Classification]

F
-->G[📂 Hierarchical Storage]
```

---

# 🚀 Features

### 🎯 Layout-Aware Document Parsing

Deep-learning based region localization using YOLO.

---

### 🧠 Contextual Semantic Understanding

Gemini Vision performs multimodal reasoning on cropped regions.

---

### 🔥 Multi-Recipient Support

Single document → Multiple folders

```text
Chief Engineer
Chairman
```

↓

```text
classified/
├── chief engineer/
└── chairman/
```

---

### ⚙ Dynamic Category Generation

Folders are automatically created:

* Chief Engineer
* Chairman
* Secretary
* Unknown

---

# 🏗 Architecture

```text
               Input Document
                      │
                      ▼
      ┌────────────────────────────┐
      │ YOLO Document Layout Model │
      └────────────────────────────┘
                      │
                      ▼
              Region Cropping
                      │
                      ▼
      ┌────────────────────────────┐
      │ Gemini Vision Inference    │
      └────────────────────────────┘
                      │
                      ▼
         Receiver Designation Extraction
                      │
                      ▼
           Multi-label Classification
                      │
                      ▼
            Hierarchical File Storage
```

---

# 📁 Project Structure

```text
KSHB-Project
│
├── app.py
├── cropper.py
├── classifier.py
├── gemini_extractor.py
│
├── models
│     └── dla-model.pt
│
├── temp
│
└── classified
      ├── chief engineer
      ├── chairman
      ├── secretary
      └── unknown
```

---

# 🛠 Tech Stack

<p align="center">

<img src="https://skillicons.dev/icons?i=python,vscode,opencv,git,github"/>

</p>

---

# 📈 Workflow

```mermaid
graph LR

A[Upload]
-->B[Detection]

B
-->C[Crop]

C
-->D[Gemini Vision]

D
-->E[Classification]

E
-->F[Storage]
```

---

# ⚡ Run

```bash
pip install -r requirements.txt
python app.py
```

---

# 🔮 Future Enhancements

* OCR-Free Document Intelligence
* LayoutLM Integration
* Batch Processing
* Semantic Search
* Vector Database
* RAG Pipeline
* Metadata Indexing
* Web Deployment

---

<div align="center">

# ⭐ Star this repository if you found it useful!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00c6ff,100:0072ff&height=120&section=footer"/>

</div>
