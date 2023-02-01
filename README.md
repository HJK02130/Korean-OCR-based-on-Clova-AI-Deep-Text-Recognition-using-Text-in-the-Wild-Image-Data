# <div align=center> Korean OCR based on Clova AI Deep Text Recognition <br/> using AI Hub Data </div>

<div align=right> <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/HJK02130/Korean-OCR-based-on-Clova-AI-Deep-Text-Recognition-using-AI-Hub-Data?style=flat-square"> <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/HJK02130/Korean-OCR-based-on-Clova-AI-Deep-Text-Recognition-using-AI-Hub-Data?style=flat-square"> <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/HJK02130/Korean-OCR-based-on-Clova-AI-Deep-Text-Recognition-using-AI-Hub-Data?style=flat-square"> </div>


## Contents
1. [Overview](#overview)
2. [Issue](#issue)
3. [Environment](#environment)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Repository Explaination](#repository-explaination)
7. [Architecture](#architecture)
8. [Result](#result)
9. [Conclusion](#conclusion)
10. [Reference](#reference)
11. [Developer](#developer)


## Overview
[[DACON contest]](https://dacon.io/competitions/official/235970/overview/description)  [[Clova AI Deep Text Recognition Benchmark]](https://github.com/clovaai/deep-text-recognition-benchmark) <br/><br/>
OCR technology can greatly reduce the time and effort required to scan documents into images, simplify the operations required to convert text images into data, and increase productivity by automating the process. This project was carried out in the SW-centeral university joint AI competition <finals> conducted by DACON. We developed an 'Optical Character Recognition (OCR)' algorithm that can detect and recognize text from images in which the text part is cropped out of signboards, book covers, and sign images of Hangul text. Using the Korean text image data provided by DACON, the original Korean text image data provided by AI Hub, and the text image we created, we trained the [Korean OCR deep learning model](https://github.com/clovaai/deep-text-recognition-benchmark) provided by Clova AI, and use the accuracy as a performance evaluation metrics. However, because the PC environment was not good and the speed was slow, only 18900 of 300000 epochs, that is, 6.3% of the scheduled epochs were trained and we forced training to end before the training is completed. Also, we didn't used all data we collected. Please note this. As a result of the test, DACON's public test set showed an accuracy of 0.539, and the private test set showed an accuracy of 0.523. We believe that better results could be achieved if training was completed using all the data collected and a better GPU without hardware limitations.
<br/>
<br/>
OCR 기술은 문서를 이미지로 스캔하는 작업에 들어가는 시간과 노력을 크게 단축시킬 수 있으며, 텍스트 이미지를 데이터로 변환하는 데 필요한 운영을 간소화하고 프로세스를 자동화하여 생산성을 높일 수 있습니다. 본 프로젝트는 DACON에서 진행한 SW중심대학 공동 AI 경진대회 <본선>에서 진행된 프로젝트이며, 한글 텍스트의 간판, 책표지, 표지판 이미지 중 텍스트 부분이 crop된 이미지에서 텍스트를 탐지하고 인식할 수 있는 '광학 문자 인식(Optical Character Recognition, OCR)'을 주제로 알고리즘을 개발하였습니다. DACON에서 제공하는 한글 텍스트 이미지 데이터, AI Hub에서 제공하는 한글 텍스트 이미지 원천데이터, 그리고 KAIST에서 수집한 데이터와 직접 생성한 텍스트 이미지를 활용하여 Clova AI에서 제공하는 한글 OCR 딥러닝 모델을 학습시키고, 성능평가지표로 정확도를 사용하여 성능 평가를 진행하였습니다. 단, PC 환경이 좋지 않아 속도가 느린 관계로 예정되었던 Epoch 300000 중 18900 즉, 예정 Epoch의 6.3%만 학습시키고 강제로 학습을 종료하였고, 수집한 모든 데이터를 사용하지 못하였습니다. 이 점 참고해주시길 바랍니다. 테스트 결과, DACON의 public test set에서는 0.539의 정확도를, private test set에서는 0.523의 정확도를 보였습니다. 비교적 성능이 좋은 GPU를 사용하여 수집 및 생성한 모든 데이터를 활용하여 하드웨어 제한 없이 학습을 완료하였다면 더 좋은 결과를 낼 수 있을 것이라고 판단합니다.

## Issue
!! We didn't used all data we collected<br/>
!! We didn't do training to the end<br/>
!! You need better GPU<br/>
!! It takes a long time

## Environment
+ Pytorch 1.3.1
+ CUDA 10.1
+ Python 3.6

## Requirements
+ lmbd
+ pillow
+ torchvision
+ nltk
+ natsort
+ tqdm

## Usage
Filetree (modifying)

## Repository Explaination
###### 📁 deep-text-recognition-benchmark<br/>code folder
> ###### 📄 json_to_txt.py<br/>Take text in info file(.json) and each image path and create a text file(.txt)
> ###### 📄 modify_txt.ipynb<br/>Modify created text file to suitable format for making lmdb data
> ###### 📄 create_lmdb_dataset.py<br/>Clova AI's deep text recognition - code creating lmdb data
> ###### 📄 train.py<br/>Clova AI deep text recognition - training code
> ###### 📄 test.py<br/>Clova AI deep text recognition - test code

###### 📁 saved_models<br/> trained model
> ###### 📁 None-VGG-BiLSTM-CTC-Seed1111<br/>We tried to test model NVBC(None-VGG-BiLSTM-CTC)
>> ###### 📄 best_norm_ED.pth<br/>The model we trained that have best norm ED
>> ###### 📄 best_accuracy.pth<br/>The model we trained that have best accuracy

###### 📁 result<br/>lmdb data
> ###### 📁 trainig<br/>training lmdb data(empty)
> ###### 📁 validatdion<br/>validation lmdb data(empty)
<br/>

## Architecture
### Training Data
+ [DACON training data](https://dacon.io/competitions/official/235970/overview/description)
	+ Image : Text cropped text-in-the-wild image data
	+ Text file(.csv) : Text in image
+ [AI Hub data](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=105)
	+ Image : Text-in-the-wild image
	+ Info file(.json) : Text, text bounding box coordinate and etc.
+ Generated data
	+ We made new generated data with random Korean text image with random font, text impact and random background image
	+ [generator reference](https://github.com/Belval/TextRecognitionDataGenerator)
+ [Kaist data](http://www.iapr-tc11.org/mediawiki/index.php/KAIST_Scene_Text_Database)
	+ Text-in-the-wild image data
	
<br/>

### Preprocessing
#### Image preprocessing
+ AI Hub data 
	+ Create text cropped image using bounding box coordinate in info file
#### File preprocessing
+ DACON data
	+ Take text in text file(.csv) and each image path and create a text file(.txt)
+ AI Hub data
	+ Take text in info file(.json) and each image path and create a text file(.txt)
+ Generated data
	+ Create text file with each image path and each text of image
#### Create LMDB data
~~~
python deep-text-recognition-benchmark/create_lmdb_dataset.py \
--inputPath train \
--gtFile train/gt.txt \
--outputPath result/training
~~~

#### Split data
+ split training data to training, validation data
+ [DACON test data](https://dacon.io/competitions/official/235970/data)

<br/><br/>

### Training
We tried to train model NVBC(None-VGG-BiLSTM-CTC) and used pre-trained OCR model
~~~
python deep-text-recognition-benchmark/train.py \
--train_data result/training \
--valid_data result/validation \
--input_channel 1 \
--output_channel 256 \
--hidden_size 256 \
--Transformation None \
--FeatureExtraction VGG \
--SequenceModeling BiLSTM \
--Prediction CTC \
—saved_model korean_g2.pth
--character " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_\`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘" —FT
~~~

<br/>

### Test
~~~
python deep-text-recognition-benchmark/test.py \
--eval_data result/validation --benchmark_all_eval \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC \
--saved_model saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth
~~~

## Result
!! We didn't train model to the end !!
||Public test set|Private test set|
|:---:|:---|:---|
|Accuracy|0.53909|0.52316|

If you have better GPU and lots of time to train, you will undoubtedly be able to achieve near-perfect accuracy.

## Conclusion
Using Clova AI text recognition with deep learning methods in this DACON contest, It was able to learn a wide range of data handling methods by performing various data processing such as image preprocessing, image generating, and lmdb conversion of various text-in-the-wild image and other documents. In addition, it was a valuable opportunity to improve image and natural language problem solving skills by training with tuning, analyzing and correcting problems in the results. However, due to hardware limitations, all the collected data could not be used, and even training had to be forcibly terminated, resulting in lower accuracy than expected. We are sure that better results would be obtained if all epochs were completely trained using all the data in a better hardware environment, and We are planning to train our trained model in better hardware environment. If Korean OCR using text-in-the-wild develops further, the time and effort to detect and recognize text in images will be greatly reduced, and productivity can be increased by minimizing the process required to convert text images into data and automating the process.

## Reference


## Developer
Hyunji Kim, Yeaji Kim, Changhyeon Lee.
<br />
Hyunji Kim <a href="mailto:hjk021@khu.ac.kr"> <img src ="https://img.shields.io/badge/Gmail-EA4335.svg?&style=flat-squar&logo=Gmail&logoColor=white"/> 
[<img src="https://img.shields.io/badge/Notion-000000?style=flat-square&logo=Notion&logoColor=white"/>](https://read-me.notion.site/Hyunji-Kim-9dbdb62cc84347feb85b3c58225bb63b)
	<a href = "https://github.com/HJK02130"> <img src ="https://img.shields.io/badge/Github-181717.svg?&style=flat-squar&logo=Github&logoColor=white"/> </a>
