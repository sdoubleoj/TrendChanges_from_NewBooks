from webdriver_manager.chrome import ChromeDriverManager # 자동으로 크롬드라이버(가상브라우저) 파일을 다운로드해주는 라이브러리
from selenium.webdriver.chrome.service import Service # 다운로드된 크롬드라이버 파일을 연결하기 위해 활용
from selenium.webdriver.support.select import Select
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup 
import time
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")
import os
import platform
import re
from glob import glob
from urllib.request import urlopen
import requests
from tqdm import tqdm
from wordcloud import WordCloud 
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import font_manager, rc
import platform
import seaborn as sns
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from collections import Counter
import IPython
import IPython.display


# 카테고리 선택
def select_info():    
    # 카테고리 설정
    while True:
        print("카테고리 : 분야 종합, 소설, 시/에세이, 건강, 경제/경영, 정치/사회, 역사/문화, 예술/대중문화, 과학, 여행, 청소년, 유아(0~7세), 어린이(초등), 만화, 컴퓨터/IT")
        categories = ['분야 종합', '소설', '시/에세이', '건강', '경제/경영', '정치/사회', '역사/문화', '예술/대중문화', '과학', '여행', '청소년', 
                      '유아(0~7세)', '어린이(초등)', '만화','컴퓨터/IT']
        category = input("카테고리를 작성해주세요: ")

        if category in categories:
            break
        else:
            print("카테고리에 없는 분야를 입력하셨습니다")
            
    
    # 년 / 월 선택
    current_year = datetime.datetime.today().year
    current_month = datetime.datetime.today().month
    
    while True:
        # 년도 입력, 잘못된 양식으로 넣어도 정규화를 통해 입력가능
        searching_year = input("검색을 시작하고 싶은 년도를 입력해주세요(2017~): \'예시 2022\' ")
        searching_year = re.sub(r"[^0-9]","",searching_year)
        
        # 2017년부터 현재년도까지만 입력 가능
        if int(searching_year) >= 2017 and int(searching_year) <= current_year:
            break
        else:
            print("2017년부터 현재년도까지만 검색이 가능합니다")
    
    while True:
        # 월 입력, 잘못된 양식으로 넣어도 정규화를 통해 입력가능
        searching_month = input("검색을 시작하고 싶은 월을 입력해주세요: '예시 03' ")
        searching_month = re.sub(r"[^0-9]","",searching_month)
        
        # 현재년도에는 선택한 달이 이번 달보다 크면 입력 불가능
        if int(searching_year) == current_year and int(searching_month) > current_month:
            print("현재 날짜보다 이후는 검색이 불가합니다.")
        else:
            break
    
    return category, searching_year, searching_month


# 지정년도/월/주 별 시간도서 엑셀 파일 다운로드 및 데이터프레임 처리
def make_df():
    # select_info() 함수를 통해 category, searching_year, searching_month 생성
    category, searching_year, searching_month = select_info()
    
    # 오늘의 년 / 월
    current_year = datetime.datetime.today().year
    current_month = datetime.datetime.today().month
    
    # 지정한 년 / 월 이전 데이터들을 제거하기 위해 설정
    df_year, df_month = int(searching_year), int(searching_month)
    
    # 크롬 드라이버 생성
    service = Service(executable_path=ChromeDriverManager().install())
    my_system = platform.system()
    
    # 카테고리 특수문자 제거
    category_n = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ]","",category) 
    
    #os별 파일 저장 경로 지정
    if my_system == 'Windows':
        download_path = os.getcwd() + '\\books_excel_'+category_n+"_"+searching_year+"_"+searching_month
    elif my_system == 'Darwin':
        download_path = os.getcwd() + '/books_excel_'+category_n+"_"+searching_year+"_"+searching_month
    
    download_folder = './books_excel_'+category_n+"_"+searching_year+"_"+searching_month
    
    if my_system == 'Windows':
        folder = '.\\books_excel_'+category_n+"_"+searching_year+"_"+searching_month+"\\*"
    elif my_system == 'Darwin':
        folder = './books_excel_'+category_n+"_"+searching_year+"_"+searching_month+"/*"
    
    if not os.path.isdir(download_folder):
        os.mkdir(download_folder)

    searching_year = int(searching_year)
    searching_month = int(searching_month)

    prefs = {'savefile.default_directory' : download_path,'download.default_directory' : download_path}
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(service=service, options = chrome_options)



    # url 설정
    newbooks_url = 'http://www.kyobobook.co.kr/newproduct/newProductList.laf?orderClick=Ca1'
    # url 열기
    driver.get(newbooks_url)
    
    category_click = 0
    while searching_year<= current_year:
        # 년도 선택
        year_select = Select(driver.find_element_by_name('yyyy'))
        year_select.select_by_visible_text(str(searching_year)+'년')


        # 월 별 반복
        for searching_month_idx in range(searching_month-1, 12):
            # 미래에 대한 검색 방지
            if (searching_year == current_year) and (searching_month_idx+1 >= current_month):
                break
            

            # 월 선택
            month_select = Select(driver.find_element_by_name('mm'))
            month_select.select_by_index(searching_month_idx)

            # 주 별 반복
            for searching_week_idx in range(6):
                # 주 선택
                week_select = Select(driver.find_element_by_name('week'))
                week_select.select_by_index(searching_week_idx)

                # 예외 처리(검색 결과가 없는 경우)
                try:
                    # 클릭 버튼 누르기
                    search_button = '/html/body/div/div[1]/div[2]/form/div/div[3]/div[3]/div/div[1]/div/a'
                    driver.find_element_by_xpath(search_button).click()

                    # 카테고리 선택
                    # 카테고리 선택이 되면 category_click을 1로 바꿔 크롤링 할때마다 클릭하는 것을 방지
                    if category_click == 0:
                        driver.find_element_by_link_text(category).click()
                        category_click = 1
                    else:
                        pass
                    # 엑셀 다운로드 받기
                    download_excel = '/html/body/div/div[1]/div[2]/form/div/div[3]/div[3]/div/div[2]/div[2]/a[4]'
                    driver.find_element_by_xpath(download_excel).click()


                    print(f'{searching_year}년 {searching_month_idx+1}월 {searching_week_idx+1}주차 검색 결과 다운로드 완료.')

                    time.sleep(3)
                except:
                    print(f'{searching_year}년 {searching_month_idx+1}월 {searching_week_idx+1}주차 검색 결과가 없습니다.')
        searching_year += 1
        searching_month = 1
        
    driver.close()
    driver.quit()
    
    
    # 엑셀 파일 합쳐서 DataFrame 만들기
    df = pd.DataFrame()
    
    # 지정 폴더에 저장된 엑셀 파일명 불러오기 및 전처리 과정
    for file_name in glob("{}.xls".format(folder)):
        file = pd.read_html(file_name)[0]
        file.drop(0, inplace = True)
        file.drop(0, axis = 1, inplace = True)
        file.columns = ['ISBN','도서명','저자명','출판사명','출간일','정가']
        file.drop(1, inplace=True)
        df = pd.concat([df,file])
    df = df.drop_duplicates()
    df.정가 = df.정가.apply(lambda x: x[1:])
    df = df.sort_values(by = '출간일')
    df.reset_index(drop=True, inplace=True)
    # '출간일'이 지정한 기간 이전 데이터들은 제거
    df = df[pd.to_datetime(df.출간일) >= datetime.datetime(df_year, df_month,1)].reset_index(drop = True)
    
    return df
    




# ISBN(도서코드)를 이용해 도서별 줄거리 크롤링
def contents_crawling(df):
    print('{}개의 책이 있습니다.'.format(len(df)))
    
    singlebook_base_url = "https://www.kyobobook.co.kr/product/detailViewKor.laf?mallGb=KOR&barcode="

    failed_barcode = []
    content_list = []

    for book in tqdm(df.iterrows()):
        # 개별 책 페이지 접속 및 줄거리 크롤링
        try:
            barcode = book[1]['ISBN']
            request_url = singlebook_base_url + barcode
            web = requests.get(request_url).content
            source = BeautifulSoup(web, 'html.parser')

            content = source.find('div',{'class':'box_detail_article'}).get_text()
            content_list.append(content)

            time.sleep(3)

        # 크롤링에 실패한 경우 줄거리에 빈 문자열 넣고, 실패한 책 바코드 기록
        except: 
            barcode = book[1]['ISBN']
            failed_barcode.append(barcode)
            content_list.append("")

    df['줄거리'] = content_list

    # 크롤링에 실패한 책 재크롤링 및 데이터 끼워넣기
    retry = 0
    while True:
        none_index = df[df.줄거리 == ''].index
        for i in tqdm(none_index):
            try:
                barcode = df.at[i,'ISBN']
                request_url = singlebook_base_url + barcode
                web = requests.get(request_url).content
                source = BeautifulSoup(web, 'html.parser')
                content = source.find('div',{'class':'box_detail_article'}).get_text()
                df.at[i,'줄거리'] = content
                time.sleep(3)
            except:
                print('{} 의 줄거리를 가져오지 못했습니다.'.format(df.at[i,'도서명']))
                pass
        if (len(none_index) == 0) or (retry == 3):
            break
        retry += 1


    # 크롤링해온 줄거리 텍스트에서 필요없는 공백 제거
    df.줄거리 = df.줄거리.apply(lambda x : x.replace("\t",''))
    df.줄거리 = df.줄거리.apply(lambda x : x.replace("\n",''))
    df.줄거리 = df.줄거리.apply(lambda x : x.replace("\r",''))

    # 도서별 출간년도/월 칼럼 추가
    df['year'] = df.출간일.apply(lambda x : int(str(x)[:4]))
    df['month'] = df.출간일.apply(lambda x : int(str(x)[4:6]))

    return df


# 출간년도/월별 말뭉치 합친 데이터프레임 생성
def make_corpus_bydate(df_book):
    df_corpus_bydate = pd.DataFrame(columns=['date','corpus'])
    
    start = df_book['year'].min()
    finish = df_book['year'].max()+1
    
    # 'year'와 'month'에 존재하는 경우만 줄거리 말뭉치 생성
    for year in tqdm(range(start, finish)):
        for month in range(1,13):
            corpus_list = []
            date = datetime.date(year, month, 1)
            
            # 해당하는 줄거리 없으면 말뭉치 리스트에 공백으로 추가
            for book in df_book.iterrows():
                try:
                    if (book[1]['year'] == year) and (book[1]['month'] == month):
                        corpus_list.append(str(book[1]['줄거리']))
                except:
                    pass
            corpus = " ".join(corpus_list)
            if corpus != "":
                new_corpus = { 'date' : date, 'corpus' : corpus}
                df_corpus_bydate = df_corpus_bydate.append(new_corpus,ignore_index=True)
    return df_corpus_bydate



def make_word_dic(sentence):

    
    # 많이 나오지만 분석에 도움이 되지 않는 단어들 제거
    del_list = ['같다','모든','하다', '있다', '되다', '이다', '돼다', '않다', '그렇다', '아니다', '이렇다', '그렇다', '어떻다',
               '우리','저자','통해','무엇','대한','대해','위해','또한','이야기','지금','모두','리즘','모습','먼저','이제','제대로',
               '얼마나','바로','이후','여러','누구','불구','동안','크게','서도','감히','로서','달리','만큼','비롯','매우','가까이',
               '미리','어쩌면','면서','더구나','오히려','게다가','관련','어디','역시','더욱','여기','저기','거기','결코','거나','앞서',
               '마치','경우','사이'] 

    
    # kernel dead가 뜨면 이모티콘 제거 해야함
    tokenizer = Okt()
    raw_pos_tagged = tokenizer.pos(sentence, norm=True, stem=True) # POS Tagging
    word_cleaned = []
    
    # 명사와 알파벳를 제외한 나머지 단어들은 제거 
    for word in raw_pos_tagged: #  ('서울', 'Noun'),
        if word[1] in ['Noun','Alpha']: # Foreign ==", " 와 같이 제외되어야할 항목들
            if (len(word[0]) != 1) & (word[0] not in del_list): # 한 글자로 이뤄진 단어들을 제외 & 원치 않는 단어들을 제외
                word_cleaned.append(word[0])
    
    # 결과를 dict 형태로 저장
    result = Counter(word_cleaned)
    word_dic = dict(result.most_common())
    return word_dic






def visualization(df):
    # 윈도우, 맥에 따라 다른 코드 적용을 위해 내 운영체제 확인
    my_system = platform.system()

    df.rename(columns={0:'corpus'},inplace=True)
    corpus = df['corpus'].tolist()
    corpus = ' '.join(corpus)
    
    # 해당 기간동안 가장 많이 나온 단어 5개 선정
    word_dic = make_word_dic(corpus)
    top_5 = sorted(word_dic.items(), key=lambda x: (x[1]), reverse=True)[:5]
    top_5_words = []
    for word in top_5:
        top_5_words.append(word[0])

    # 각각의 corpus의 단어 수들을 word_counts에 담음
    word_counts = []
    for i in range(len(df)):
        word_count = make_word_dic(df.iloc[i,1]) 
        word_counts.append(word_count)

    # top 5 단어에 대하여 기간별로 나온 횟수를 카운팅해서 데이터프레임 형태로 저장
    words_trend = {}
    for word in top_5_words: 
        count_trend = {}
        for i in range(len(word_counts)):
            count_trend[df.iloc[i,0]]=word_counts[i].get(word, 0)
        words_trend[word] = count_trend.values()
    word_trend = pd.DataFrame(data = words_trend.values(),columns=df.date, index = words_trend.keys())
    word_trend = word_trend.T    
    
        
    #워드클라우드 객체 생성
    if my_system == 'Windows':
        word_cloud = WordCloud(font_path="C:/Windows/Fonts/malgun.ttf", 
                       max_words=50,
                       width=2000, height=1000, prefer_horizontal= 1.0,
                       background_color='white').generate_from_frequencies(word_dic)
    elif my_system == 'Darwin':
        word_cloud = WordCloud(font_path="/Library/Fonts/AppleGothic.ttf", 
                       max_words=50,
                       width=2000, height=1000, prefer_horizontal= 1.0,
                       background_color='white').generate_from_frequencies(word_dic)


    if my_system == 'Windows':
        plt.rc("font", family="Malgun Gothic")
    elif my_system == 'Darwin':
        plt.rc("font", family="AppleGothic")
    
    IPython.display.clear_output()
    # 워드클라우드 그리기
    plt.figure(figsize=(10,10)) 
    plt.imshow(word_cloud)
    plt.axis("off") #축 제거
    plt.tight_layout(pad=0) #레이아웃 여백 제거
    plt.show()
    
    # top 5 추세 그래프 그리기
    plt.figure(figsize = (15,7))
    sns.lineplot(data=word_trend)  # label 설정값을 legend에 나타날 수 있음
    plt.legend()
    plt.xlabel("날짜")
    plt.ylabel("개수")    
    plt.title("Top5 단어 발생 추이", fontsize = 18)

    plt.show()


# 모든 함수 한번에 실행
def book_trend():
    df =make_df()
    df_book = contents_crawling(df)
    df_corpus_bydate = make_corpus_bydate(df_book)
    visualization(df_corpus_bydate)
    
    return df_corpus_bydate

    

# tf-idf 로 키워드 추출하는 함수
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
def tfidfscore_by_word(df, col_name, top_n ):

    corpus_all =list(df[col_name].dropna())
    corpus_all_nouns = []

    for corpus in tqdm(corpus_all):
        tokenizer = Okt()
        raw_pos_tagged = tokenizer.pos(corpus, norm=True, stem=True)
        del_list = ['같다','모든','하다', '있다', '되다', '이다', '돼다', '않다', '그렇다', '아니다', '이렇다', '그렇다', '어떻다',
                   '우리','저자','통해','무엇','대한','대해','위해','또한','이야기','지금','모두','리즘','모습','먼저','이제','제대로',
                   '얼마나','바로','이후','여러','누구','불구','동안','크게','서도','감히','로서','달리','만큼','비롯','매우','가까이',
                   '미리','어쩌면','면서','더구나','오히려','게다가','관련','어디','역시','더욱','여기','저기','거기','결코','거나','앞서',
                   '마치','경우','사이'] 

        word_cleaned = []

        for word in raw_pos_tagged: 
            if word[1] in ['Noun','Alpha']: # 명사, 영어단어만 남김
                if (len(word[0]) != 1) & (word[0] not in del_list): # 한 글자로 이뤄진 단어들을 제외 & 원치 않는 단어들을 제외
                    word_cleaned.append(word[0])
        corpus_new = " ".join(word_cleaned)
        corpus_all_nouns.append(corpus_new)

    # TF-IDF 계산: 특정 문서가 아닌 대부분의 문서에서 발견되는 단어는 중요도를 낮춤
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(corpus_all_nouns)
    tfidf =  pd.DataFrame(tfidf.toarray(),columns = vect.get_feature_names())
    tfidf = pd.DataFrame(tfidf.sum(),columns=['tfidf'])

    tfidf = tfidf.sort_values(by='tfidf',ascending=False)[:top_n]

    return tfidf


# n-gram
#from sentence_transformers import SentenceTransformer
def n_gram_keywords(doc, n_gram_range, top_n_words, n_candidates):

    okt = Okt()
    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens') # SBERT 모델 가져오기

    tokenized_doc = okt.pos(doc) # 토크나이징
    # 1글자 이상 명사만 추출
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if (word[1] in ['Noun','Alpha']) and (len(word[0])>1)]) 
    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns]) # n-gram 범위대로 단어 벡터화
    candidates = count.get_feature_names()

    doc_embedding = model.encode([doc]) #SBERT 모델에 문서 임베딩 ()
    candidate_embeddings = model.encode(candidates) #SBERT 모델에 n-gram 단어 임베딩 ()

    print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=top_n_words, nr_candidates=n_candidates))

    return