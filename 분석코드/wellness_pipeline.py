import pandas as pd
import re



from kiwipiepy import Kiwi
from soynlp.normalizer import repeat_normalize

# 1. 파일 불러오기 (.xlsx 파일이므로 read_excel 사용)
file_path = r'C:\Users\user\Desktop\세미2\데이터 - 웰니스 대화 스크립트-20260415T032342Z-3-001\데이터 - 웰니스 대화 스크립트\02)웰니스_대화_스크립트_데이터셋.xlsx'

try:
    df = pd.read_excel(file_path)
    print("파일 로드 성공!")
except Exception as e:
    print(f"파일 로드 실패: {e}")

# 2. 병합된 셀 채우기 및 라벨링
df['intent'] = df['intent'].ffill()

mapping = {
    '정신증상/우울감': 'Sadness', '정신증상/슬픔': 'Sadness', '정신증상/불안': 'Sadness',
    '정신증상/식욕저하': 'Sadness', '정신증상/식욕증가': 'Sadness', '정신증상/불면': 'Sadness',
    '정신증상/외로움': 'Loneliness',
    '정신증상/분노': 'Anger', '정신증상/감정조절이상': 'Anger', '정신증상/초조함': 'Anger',
    '정신증상/무기력': 'Worthlessness', '정신증상/피로': 'Worthlessness', 
    '정신증상/죄책감': 'Worthlessness', '정신증상/자신감저하': 'Worthlessness', 
    '정신증상/자존감저하': 'Worthlessness',
    '정신증상/집중력저하': 'Cognitive Dysfunction',
    '정신증상/절망감': 'Hopelessness',
    '정신증상/상실감': 'Emptiness',
    '정신증상/자살충동': 'Suicide Intent'
}
df['label'] = df['intent'].map(mapping)

# [1단계] 텍스트 정제
def clean_text(text):
    if not isinstance(text, str): return ""
    # 특수문자 정제 (?, ! 유지)
    text = re.sub(r'[^가-힣a-zA-Z0-9?!., ]', '', text)
    # 반복 문자 정규화
    text = repeat_normalize(text, num_repeats=2)
    return text.strip()

# [2단계] 토큰화 (Kiwi 사용)
kiwi = Kiwi()

def tokenize_kiwi(text):
    # 형태소 분석
    result = kiwi.analyze(text)
    # 일반명사(NNG), 고유명사(NNP), 동사(VV), 형용사(VA) 위주 추출
    tokens = []
    for res in result:
        for word, tag, _, _ in res[0]:
            if tag in ['NNG', 'NNP', 'VV', 'VA', 'XR', 'MAG']:
                tokens.append(word)
    return " ".join(tokens)

# 전처리 적용
print("데이터 전처리를 시작합니다. 잠시만 기다려 주세요...")
df['cleaned_utterance'] = df['utterance'].apply(clean_text)
df['tokenized_utterance'] = df['cleaned_utterance'].apply(tokenize_kiwi)

# 3. 결과 확인 및 저장
print("--- 전처리 완료 샘플 ---")
print(df[['utterance', 'label', 'tokenized_utterance']].head())

df.to_csv('wellness_python_only_final.csv', index=False, encoding='utf-8-sig')
print("\n전처리 완료! 'wellness_python_only_final.csv' 파일이 생성되었습니다.")