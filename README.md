# [아기돌고래]ECG-classification-HDAI-2021

## Model Architecture

<img src="./[주제2. 아기돌고래]/img/캡처1.PNG" width="500">

> 모델 전체 구조입니다.

## Requirement

### Dataset

원래 제공받은 xml 데이터는 12 leads와 각 전극에서의 median 데이터가 있었으나, 저희는 공통된 <mark> 8개의 leads </mark> **[rhythm_I, rhythm_II, rhythm_V1, rhythm_V2, rhythm_V3, rhythm_V4, rhythm_V5, rhythm_V6]** 만 가지고 분류 예측 모델의 입력으로 활용하였습니다. 위와 같이 선별한 8개의 time series leads feature들에 아래와 같은 denoise preprocess를 수행해주었습니다.

1. **ffill**로 신호 길이가 5천이 아닌 예외적인 신호들의 경우 주변 신호들로 마지막 결측 신호 보완
2. **bandpass**로 0.67 ~ 15 hz의 신호만 통과
3. 소수점 첫 번째 자리까지 **split**

위와 같이 전처리를 수행한 데이터에 최종적으로 **맨 마지막 열**에 해당 신호들이 정상(0)인지 부정맥(1)인지에 대한 라벨을 아래 첨부한 그림과 같이 추가하였습니다.
원래는 부정맥 라벨로 1에서 15까지의 각각의 class에 대한 라벨링을 진행하여 총 16개의 classification을 수행하려 하였으나, 본 대회에서 **binary classification** 으로 최종 예측을 진행하기 때문에 15개의 부정맥 라벨은 모두 1로 처리하였습니다.

<!-- <img src="https://github.com/17011813/ECG-classification-HDAI-2021/blob/main/%EB%9D%BC%EB%B2%A8.PNG?raw=true" width="1000"> -->

> 하나의 부정맥 데이터의 구성 예시입니다.


### Software

- Python 3.8.11
- Matplotlib 3.5.0
- Numpy 1.20.3
- Pandas 1.3.3
- PyTorch 1.9.1
- Scikit-learn 1.0.1
- Scipy 1.7.3
- Tqdm 4.62.3
- xmltodict 0.12.0

## Run

### Test Predict

test set 성능 평가 실행을 위해 보내드린 아기돌고래 폴더내의 electrocardiogram/data/test/ 폴더에 들어가신후, 기존에 제공해주셨던 validation 폴더와 같이 normal / arrhythmia 두 개의 폴더 안에 각각 정상과 비정상 test 데이터를 넣어주시고 **main.py** 코드를 실행하시면 AUC score를 확인하실 수 있습니다.

만약 제공해주셨던 validation 데이터와 다르게 정상 / 부정맥 폴더 구분 없이 하나의 폴더안에 모든 test 데이터가 들어가 있다면 **main.py**에서 120, 121번째 줄을 주석 처리해주시고, 123번째 줄의 주석을 해지하시고 돌리시면 하나의 폴더에서 모든 test 데이터를 가져와서 실행됩니다.

```sh
$ python main.py
```


코드 실행 중에 에러가 발생하셨다면 아래의 연락처로 언제든지 편하게 연락 주시면 빠르게 답변 드리도록 하겠습니다.

**정원용**: justin715@korea.ac.kr (010-7370-5732)

**김동준**: djkim0516@gmail.com (010-3722-3683)

**노윤아**: yoona028@korea.ac.kr (010-7328-8238)

좋은 대회 열어주셔서 진심으로 감사합니다.
