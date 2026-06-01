import pandas as pd

# 1. 재료 데이터 로드
# 우울 관련 데이터: 19,666행 (사용자 지정 파일)
wellness = pd.read_csv('cleaned_wellness.csv') 

# 주제별 일상 대화 데이터 (주 원천)
subject = pd.read_csv('cleaned_subject.csv')

# 챗봇 데이터 (5,261행 전체)
chatbot = pd.read_csv('cleaned_chatbot.csv') 

# 2. 일상 데이터 통합 풀(Pool) 생성 (DS1용)
daily_pool = pd.concat([subject, chatbot], ignore_index=True)

print("--- 데이터 로드 및 확인 ---")
print(f"우울 데이터(Wellness): {len(wellness)}행")
print(f"일상 데이터 풀(Subject + Chatbot): {len(daily_pool)}행")
print(f"챗봇 데이터(Chatbot): {len(chatbot)}행\n")

# 데이터 섞기 및 저장 함수
def finalize_and_save(df, file_name):
    # 우울과 일상이 섞이도록 전체 셔플
    final_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # 한글 깨짐 방지 인코딩 저장
    final_df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f":white_check_mark: {file_name} 생성 완료 (총 {len(final_df)}행)")
    return final_df

# ---------------------------------------------------------
# 3. 논문 기준에 따른 4가지 데이터세트 구축
# ---------------------------------------------------------

print("--- 실험용 데이터세트 구축 시작 ---")

# [Dataset 1]: 우울(19,666) + 일상(전체 풀에서 23,000개 샘플링)
daily_ds1 = daily_pool.sample(n=23000, random_state=42)
ds1 = finalize_and_save(pd.concat([wellness, daily_ds1]), 'dataset_1.csv')

# [Dataset 2]: 우울(19,666) + 일상(주제별 데이터에서 18,000개 샘플링)
# ※ 논문 설명: DS1에서 챗봇을 제외한 구성 (비율 1:1 맞춤)
daily_ds2 = subject.sample(n=18000, random_state=42)
ds2 = finalize_and_save(pd.concat([wellness, daily_ds2]), 'dataset_2.csv')

# [Dataset 3]: 우울(19,666) + 일상(주제별 데이터에서 1,000개 샘플링)
daily_ds3 = subject.sample(n=1000, random_state=42)
ds3 = finalize_and_save(pd.concat([wellness, daily_ds3]), 'dataset_3.csv')

# [Dataset 4]: 우울(19,666) + 일상(DS3의 일상 1,000개 + 챗봇 5,261개 전체)
# ※ 논문 설명: DS3에 챗봇 데이터를 추가하여 일상 인텐트 평균 비율 1:1 맞춤
daily_ds4 = pd.concat([daily_ds3, chatbot], ignore_index=True)
ds4 = finalize_and_save(pd.concat([wellness, daily_ds4]), 'dataset_4.csv')

print("\n:sparkles: 모든 데이터세트가 논문의 의도대로 구축되었습니다.")