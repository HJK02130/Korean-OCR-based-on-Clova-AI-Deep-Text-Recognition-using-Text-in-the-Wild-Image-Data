# <div align=center> Korean OCR based on Clova AI Deep Text Recognition <br/> using AI Hub Data </div>

<div align=right> <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/HJK02130/Korean-OCR-based-on-Clova-AI-Deep-Text-Recognition-using-AI-Hub-Data?style=flat-square"> <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/HJK02130/Korean-OCR-based-on-Clova-AI-Deep-Text-Recognition-using-AI-Hub-Data?style=flat-square"> <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/HJK02130/Korean-OCR-based-on-Clova-AI-Deep-Text-Recognition-using-AI-Hub-Data?style=flat-square"> </div>


### Contents
1. [Overview](#overview)
2. [Environment](#Environment)
3. [Languages and Development Tools](#languages-and-development-tools)
4. [Usage](#usage)
5. [Architecture](#architecture)
6. [Result](#result)
7. [Conclusion](#conclusion)
8. [Reference](#reference)
9. [Developer](#developer)


### Overview
[[DACON contest]](https://dacon.io/competitions/official/235970/overview/description)<br/>

<br/>
<br/>
OCR 기술은 문서를 이미지로 스캔하는 작업에 들어가는 시간과 노력을 크게 단축시킬 수 있으며, 텍스트 이미지를 데이터로 변환하는 데 필요한 운영을 간소화하고 프로세스를 자동화하여 생산성을 높일 수 있습니다. 본 프로젝트는 DACON에서 진행한 SW중심대학 공동 AI 경진대회 <본선>에서 진행된 프로젝트이며, 한글 텍스트의 간판, 책표지, 표지판 이미지 중 텍스트 부분이 crop된 이미지에서 텍스트를 탐지하고 인식할 수 있는 '광학 문자 인식(Optical Character Recognition, OCR)'을 주제로 알고리즘을 개발하였습니다. DACON에서 제공하는 한글 텍스트 이미지 데이터, AI Hub에서 제공하는 한글 텍스트 이미지 원천데이터, 그리고 KAIST에서 수집한 데이터와 직접 생성한 텍스트 이미지를 활용하여 Clova AI에서 제공하는 한글 OCR 딥러닝 모델을 학습시키고, 성능평가지표로 정확도를 사용하여 성능 평가를 진행하였습니다. 단, PC 환경이 좋지 않아 속도가 느린 관계로 예정되었던 Epoch 300000 중 18900 즉, 예정 Epoch의 6.3%만 학습시키고 강제로 학습을 종료하였고, 수집한 모든 데이터를 사용하지 못하였습니다. 이 점 참고해주시길 바랍니다. 테스트 결과, DACON의 public test set에서는 0.539의 정확도를, private test set에서는 0.523의 정확도를 보였습니다. 비교적 성능이 좋은 GPU를 사용하여 수집 및 생성한 모든 데이터를 활용하여 속도 및 Epoch 제한 없이 학습을 완료하였다면 더 좋은 결과를 낼 수 있을 것이라고 판단합니다.

### Environment
+ Python 3.6

### Languages and Development Tools
<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=GoogleColab&logoColor=white"/>

### Usage
Filetree (modifying)

### Architecture
+ 
		

### Result
||Public test set|Private test set|
|:---:|:---|:---|
|Accuracy|0.89935|0.89578|

ddd

### Conclusion


### Reference


### Developer
Hyunji Kim, Yeaji Kim, Changhyeon Lee.
<br />
Hyunji Kim <a href="mailto:hjk021@khu.ac.kr"> <img src ="https://img.shields.io/badge/Gmail-EA4335.svg?&style=flat-squar&logo=Gmail&logoColor=white"/> 
[<img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=Notion&logoColor=white"/>](https://read-me.notion.site/Hyunji-Kim-9dbdb62cc84347feb85b3c58225bb63b)
	<a href = "https://github.com/HJK02130"> <img src ="https://img.shields.io/badge/Github-181717.svg?&style=flat-squar&logo=Github&logoColor=white"/> </a>
