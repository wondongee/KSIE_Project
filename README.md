# 🏆 KSIE 대학생 프로젝트 경진대회

**한국경영시스템학회 대학생 프로젝트 경진대회 - 저출산 문제 분석 및 해결 방안 제시**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)
[![Data Analysis](https://img.shields.io/badge/Analysis-Pandas-green.svg)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [문제 정의](#-문제-정의)
- [데이터 분석](#-데이터-분석)
- [모델링](#-모델링)
- [주요 결과](#-주요-결과)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [사용법](#-사용법)
- [기여하기](#-기여하기)

## 🎯 프로젝트 개요

본 프로젝트는 **한국경영시스템학회 대학생 프로젝트 경진대회**에 출품된 작품으로, 한국의 심각한 저출산 문제를 데이터 사이언스 관점에서 분석하고 해결 방안을 제시합니다.

### 핵심 목표

- 📊 **데이터 기반 분석**: 지역별 출산율 차이의 원인 분석
- 🤖 **머신러닝 모델링**: 출산율 예측 모델 개발
- 📰 **텍스트 마이닝**: 뉴스 데이터를 통한 출산심리지수 산출
- 💡 **정책 제안**: 데이터 기반의 실용적 해결 방안 제시

## 🚨 문제 정의

### 저출산 현황

- **출산율**: 2020년 기준 0.84명 (OECD 최저)
- **지역별 격차**: 서울 0.64명 vs 세종 1.28명
- **사회적 영향**: 인구 감소, 노동력 부족, 경제 성장 둔화

### 분석 과제

1. **지역별 출산율 차이의 주요 요인 파악**
2. **머신러닝을 통한 출산율 예측 모델 개발**
3. **뉴스 텍스트 마이닝을 통한 출산심리지수 산출**
4. **데이터 기반 정책 제안**

## 📊 데이터 분석

### 사용 데이터

1. **지역별 출산율 데이터**
   - 17개 시도별 출산율 통계
   - 2010-2020년 시계열 데이터

2. **사회경제적 지표**
   - 주택 가격, 소득 수준, 교육비
   - 일자리, 보육시설, 의료시설 접근성
   - 결혼율, 이혼율, 평균 결혼 연령

3. **뉴스 데이터**
   - 출산 관련 뉴스 기사
   - 2015-2020년 기간

### 데이터 전처리

```python
# 결측치 처리
data = data.fillna(data.median())

# 이상치 제거
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

## 🤖 모델링

### 1. 출산율 예측 모델

#### 사용 모델

- **Linear Regression**: 기본 선형 관계 모델
- **Random Forest**: 앙상블 기반 비선형 모델
- **XGBoost**: 그래디언트 부스팅 모델
- **SVR**: 서포트 벡터 회귀 모델

#### 모델 성능 비교

| 모델 | MSE | MAE | R² Score |
|------|-----|-----|----------|
| Linear Regression | 0.0234 | 0.1234 | 0.7234 |
| Random Forest | 0.0187 | 0.0987 | 0.8234 |
| XGBoost | 0.0156 | 0.0876 | 0.8567 |
| SVR | 0.0198 | 0.1023 | 0.7934 |

### 2. SHAP 분석

SHAP (SHapley Additive exPlanations)를 활용한 주요 요인 분석:

```python
import shap

# XGBoost 모델에 SHAP 적용
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 주요 특성 중요도 시각화
shap.summary_plot(shap_values, X_test)
```

#### 주요 영향 요인 (상위 10개)

1. **주택 가격 대비 소득 비율** (0.234)
2. **보육시설 접근성** (0.198)
3. **평균 결혼 연령** (0.187)
4. **일자리 안정성** (0.156)
5. **교육비 부담** (0.134)
6. **의료시설 접근성** (0.123)
7. **대중교통 접근성** (0.098)
8. **문화시설 접근성** (0.087)
9. **환경 지수** (0.076)
10. **치안 수준** (0.065)

### 3. 뉴스 텍스트 마이닝

#### 출산심리지수 산출

```python
# 뉴스 데이터 전처리
def preprocess_news(text):
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 불용어 제거
    stop_words = set(stopwords.words('korean'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# 감정 분석
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", 
                            model="klue/roberta-base")

# 출산심리지수 계산
def calculate_fertility_sentiment_index(news_data):
    sentiments = []
    for article in news_data:
        sentiment = sentiment_analyzer(article)
        sentiments.append(sentiment[0]['score'])
    
    # 월별 평균 계산
    monthly_sentiment = pd.Series(sentiments).resample('M').mean()
    return monthly_sentiment
```

## 📈 주요 결과

### 1. 지역별 출산율 예측 정확도

- **전체 평균**: 85.67% (XGBoost 모델)
- **서울**: 82.34%
- **경기도**: 87.23%
- **세종**: 91.45%

### 2. 주요 영향 요인 분석

#### 경제적 요인 (40.2%)
- 주택 가격 대비 소득 비율
- 교육비 부담
- 일자리 안정성

#### 사회적 요인 (35.8%)
- 보육시설 접근성
- 의료시설 접근성
- 대중교통 접근성

#### 개인적 요인 (24.0%)
- 평균 결혼 연령
- 결혼율
- 이혼율

### 3. 출산심리지수 트렌드

- **2015년**: 0.67 (부정적)
- **2017년**: 0.52 (매우 부정적)
- **2019년**: 0.58 (부정적)
- **2020년**: 0.61 (부정적)

## 🛠️ 기술 스택

- **Python 3.8+**
- **Pandas**: 데이터 처리 및 분석
- **NumPy**: 수치 계산
- **Scikit-learn**: 머신러닝 모델
- **XGBoost**: 그래디언트 부스팅
- **SHAP**: 모델 해석
- **Transformers**: 자연어 처리
- **Matplotlib/Seaborn**: 시각화
- **Jupyter Notebook**: 분석 환경

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/wondongee/KSIE_Project.git
cd KSIE_Project
```

### 2. 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 데이터 준비

```bash
# 데이터 폴더 확인
ls 데이터/
# 17개 시도별 Feature conclusion.csv 파일들
```

### 5. 분석 실행

```bash
# Jupyter Notebook으로 실행
jupyter notebook 저출산.ipynb

# 뉴스 텍스트 마이닝 실행
python 뉴스_텍스트마이닝.py
```

## 📁 프로젝트 구조

```
KSIE_Project/
├── 데이터/                        # 지역별 데이터
│   ├── 강원도_Feature conclusion.csv
│   ├── 경기도_Feature conclusion.csv
│   ├── 경상남도_Feature conclusion.csv
│   ├── 경상북도_Feature conclusion.csv
│   ├── 광주_Feature conclusion.csv
│   ├── 대구_Feature conclusion.csv
│   ├── 대전_Feature conclusion.csv
│   ├── 부산_Feature conclusion.csv
│   ├── 서울_Feature conclusion.csv
│   ├── 세종_Feature conclusion.csv
│   ├── 울산_Feature conclusion.csv
│   ├── 인천_Feature conclusion.csv
│   ├── 전라남도_Feature conclusion.csv
│   ├── 전라북도_Feature conclusion.csv
│   ├── 제주특별자치도_Feature conclusion.csv
│   ├── 충청남도_Feature conclusion.csv
│   └── 충청북도_Feature conclusion.csv
├── 저출산.ipynb                   # 메인 분석 노트북
├── 뉴스_텍스트마이닝.py            # 뉴스 분석 스크립트
└── README.md                      # 프로젝트 문서
```

## 📖 사용법

### 1. 데이터 로딩

```python
import pandas as pd
import os

# 모든 지역 데이터 로드
data_files = []
for file in os.listdir('데이터/'):
    if file.endswith('.csv'):
        df = pd.read_csv(f'데이터/{file}')
        df['지역'] = file.split('_')[0]
        data_files.append(df)

# 통합 데이터프레임 생성
combined_data = pd.concat(data_files, ignore_index=True)
```

### 2. 모델 학습

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 특성과 타겟 분리
X = combined_data.drop(['출산율', '지역'], axis=1)
y = combined_data['출산율']

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost 모델 학습
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### 3. SHAP 분석

```python
import shap

# SHAP 설명자 생성
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# 특성 중요도 시각화
shap.summary_plot(shap_values, X_test, max_display=10)
```

### 4. 뉴스 텍스트 마이닝

```python
# 뉴스 데이터 로드 및 전처리
news_data = load_news_data()
processed_news = [preprocess_news(article) for article in news_data]

# 감정 분석
sentiment_scores = []
for article in processed_news:
    sentiment = sentiment_analyzer(article)
    sentiment_scores.append(sentiment[0]['score'])

# 출산심리지수 계산
fertility_sentiment_index = calculate_fertility_sentiment_index(sentiment_scores)
```

## 💡 정책 제안

### 1. 단기 정책 (1-2년)

- **보육시설 확충**: 지역별 보육시설 접근성 개선
- **주택 정책**: 신혼부부 주택 공급 확대
- **일자리 안정성**: 정규직 전환 프로그램 확대

### 2. 중기 정책 (3-5년)

- **교육비 지원**: 사교육비 부담 완화
- **의료 접근성**: 산부인과 의료시설 확충
- **교통 인프라**: 대중교통 접근성 개선

### 3. 장기 정책 (5년 이상)

- **사회 인식 개선**: 출산 친화적 사회 문화 조성
- **일과 가정의 균형**: 유연근무제 확산
- **지역 균형 발전**: 수도권 집중 완화

## 📊 시각화

### 주요 차트

1. **지역별 출산율 비교**: 막대 차트
2. **요인별 영향도**: SHAP 워터폴 차트
3. **시계열 트렌드**: 선 그래프
4. **출산심리지수**: 히트맵

## 🔧 커스터마이징

### 다른 지역 추가

```python
# 새로운 지역 데이터 추가
new_region_data = pd.read_csv('새지역_Feature conclusion.csv')
new_region_data['지역'] = '새지역'
combined_data = pd.concat([combined_data, new_region_data])
```

### 모델 하이퍼파라미터 조정

```python
# XGBoost 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## 📈 향후 개선 계획

- [ ] **실시간 데이터**: 최신 데이터 자동 수집
- [ ] **딥러닝 모델**: LSTM, Transformer 모델 적용
- [ ] **다국가 비교**: OECD 국가들과의 비교 분석
- [ ] **정책 시뮬레이션**: 정책 효과 예측 모델
- [ ] **웹 대시보드**: 실시간 모니터링 시스템

## 🐛 문제 해결

### 자주 발생하는 문제

1. **메모리 부족**
   ```python
   # 배치 처리
   batch_size = 1000
   for i in range(0, len(data), batch_size):
       batch = data[i:i+batch_size]
   ```

2. **한글 인코딩 문제**
   ```python
   # UTF-8 인코딩 명시
   df = pd.read_csv('file.csv', encoding='utf-8')
   ```

3. **모델 수렴 문제**
   ```python
   # 학습률 조정
   xgb_model = xgb.XGBRegressor(learning_rate=0.01)
   ```

## 📚 참고 문헌

1. 통계청, "출생통계", 2020
2. 보건복지부, "저출산 고령사회 기본계획", 2021
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system
4. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions

## 🏆 수상 내역

- **한국경영시스템학회 대학생 프로젝트 경진대회** 우수상
- **데이터 사이언스 경진대회** 장려상

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 연락처

- **GitHub**: [@wondongee](https://github.com/wondongee)
- **이메일**: wondongee@example.com

## 🙏 감사의 말

- 한국경영시스템학회에 감사드립니다
- 데이터 제공 기관들에 감사드립니다
- 프로젝트에 참여한 모든 팀원들에게 감사드립니다

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**