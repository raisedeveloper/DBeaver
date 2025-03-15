import pandas as pd
import numpy as nppypy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 데이터셋 로드
df = pd.read_csv('dataset/StarbucksSurvey.csv')

# 범주형 'Gender' 변수 인코딩 (예: Male=0, Female=1)
# X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})  # Label Encoding 예시
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender']) 
def categorize_age(age):
    if age == 'Below 20':
        return 0  # Below 20
    elif age == 'From 20 to 29':
        return 1  # From 20 to 29
    elif age == 'From 30 to 39':
        return 2  # From 20 to 29
    else:
        return 3  # 40 and above
df['AgeCategory'] = df['Age'].apply(categorize_age)

def categorize_pur(purchase):
    if purchase == 'Coffee':
        return 0  # Below 20
    elif purchase == 'Cold drinks':
        return 1  # From 20 to 29
    elif purchase == 'Pastries':
        return 2  # From 20 to 29
    else:
        return 3  # 40 and above
df['PurchaseCategory'] = df['purchase'].apply(categorize_pur)

# 독립 변수와 종속 변수 정의
# X = df[['Gender', 'Purchase']]  # 독립 변수
X = df[['PurchaseCategory']]  # 독립 변수
y = df['AgeCategory']  # 종속 변수


# 데이터 훈련/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 초기화 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 셋에 대한 예측
y_pred = model.predict(X_test)

# R^2 점수 계산
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# 시각화
plt.figure(figsize=(15, 5))

# Gender vs Age
plt.subplot(1, 3, 1)
sns.scatterplot(x=df['PurchaseCategory'], y=df['Age'])  # 'Age' 대신 'Sales' 사용
sns.regplot(x=df['PurchaseCategory'], y=df['Age'], scatter=False, color='red')  # 회귀선
plt.xlabel('PurchaseCategory')
plt.ylabel('Age')
plt.title('PurchaseCategory vs Age')

# # Age vs Age (이건 그리기 의미가 없을 수 있지만, 예시로 두었습니다)
# plt.subplot(1, 3, 2)
# sns.scatterplot(x=df['Age'], y=df['Age'])  # 'Age' 사용
# sns.regplot(x=df['Age'], y=df['Age'], scatter=False, color='red')
# plt.xlabel('Age')
# plt.ylabel('Age')
# plt.title('Age vs Age')

# Purchase vs Age
plt.subplot(1, 3, 3)
sns.scatterplot(x=df['PurchaseCategory'], y=df['Age'])  # 'Sales' 대신 'Age' 사용
sns.regplot(x=df['PurchaseCategory'], y=df['Age'], scatter=False, color='red')  # 회귀선
plt.xlabel('PurchaseCategory')
plt.ylabel('Age')
plt.title('PurchaseCategory vs Age')

plt.tight_layout()
plt.show()

# 회귀 모델 계수 출력 (선택사항)
print("\n회귀 모델 계수:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"절편: {model.intercept_:.4f}")
