# TrendChanges_from_NewBooks

## 0. 개요
- 주제: 신간도서를 통해 알아보는 분야별 트렌드 변화
- 기간: 2022.03.31 ~ 2022.04.06 (5일)
- 참여인력: 5인
- 사용한 기술 스택
  - **Language**: Python
  - **IDE**: Jupyter Notebook
  - **Library**: Numpy, Pandas, WorldCloud, BeautifulSoup, Selenium, ChromeDriveManager, NLTK, KoNLPy


<br>

## 1. Project

최근 코로나19와 함께 재테크 붐이 일면서, 신간 및 베스트셀러 도서들의 주제가 크게 변화함을 느꼈습니다. 그 중 신간도서가 최신 이슈를 보다 빠르게 반영할 것이라고 생각하여 이를 가설로 설정한 후 확인해보고자 하였습니다.

<br>

## 2. Process

▶ 데이터 수집

  - 교보문고 홈페이지 -> '신상품' -> '화제의 신상품' 카테고리 선택 -> 기간 선택(반복) -> '엑셀로 받기'로 해당 도서들의 메타정보가 담긴 엑셀 파일 다운로드
     
  <img width="60%" alt="스크린샷 2022-05-19 오후 10 20 27" src="https://user-images.githubusercontent.com/78069770/169304900-2ab3080a-57da-4a04-8f30-5a7a2238bd34.png">

   
  - 도서 메타정보 엑셀 파일
    
  <img width="60%" alt="스크린샷 2022-05-19 오후 10 34 20" src="https://user-images.githubusercontent.com/78069770/169315179-934fb937-6827-400c-b2fa-4ad0df27d8a1.png">


  - 'ISBN' 칼럼을 이용해 각 도서별 상세페이지로 이동하여 줄거리 크롤링

  <img width="60%" alt="스크린샷 2022-05-19 오후 11 12 11" src="https://user-images.githubusercontent.com/78069770/169314628-02a966b7-a336-437f-99a0-b63872320234.png">

  <img width="721" alt="스크린샷 2022-05-19 오후 11 25 02" src="https://user-images.githubusercontent.com/78069770/169318392-f267b8bd-290b-4af4-a49e-739b2e68dcbf.png">

<br>

▶ 데이터 전처리

  - 출간일자별 도서 줄거리를 병합한 데이터프레임

  <img width="60%" alt="스크린샷 2022-05-19 오후 11 26 46" src="https://user-images.githubusercontent.com/78069770/169320774-c55f408c-751b-4222-b9fe-799312f8e82d.png">

  - 행별 단어빈도 딕셔너리 생성
  
  <img width="1004" alt="스크린샷 2022-05-19 오후 11 37 08" src="https://user-images.githubusercontent.com/78069770/169322873-6345c8b2-3392-414d-a643-6a862099b170.png">
  
<br>
  
▶ 데이터 시각화

  - 빈도 상위 n개 키워드에 대한 시각화 : line & bar chart, wordcloud
  
  ![linechart](https://user-images.githubusercontent.com/78069770/169323225-71d79993-1c0c-4ba0-9181-eea934dbbe64.png)
  ![keywordbarchart](https://user-images.githubusercontent.com/78069770/169323237-bd9cacd7-32e2-4583-8f32-3ee0d5396c5b.png)
  ![wordcloud](https://user-images.githubusercontent.com/78069770/169323242-ba07b194-fae9-4251-9cf8-e024d5b355ca.png)

<br>

▶ 전체 프로세스 함수화 (functions.py)

유저가 크롤링부터 시각화까지 일련의 과정을 거칠 수 있도록 함수화 하였습니다.

<br>

## 3. Contributions
맡은 역할 : 데이터 수집, 데이터 전처리, 시각화

<br>

## 4. Trouble Shooting

- 시점: 도서 메타정보 엑셀데이터 크롤링 단계
- 문제상황: 리스트업 해둔 카테고리 중 선택한 카테고리가 초기 페이지에 없는 경우가 있어 에러 발생
- 해결방법 1: 카테고리 선택 후 페이지 존재 여부에 따라 다음 단계로 넘어감
  -> 매 기간 선택마다 존재 여부를 확인해야하므로 작업 비효율 발생
- 해결방법 2: 별도의 변수=0을 생성. 선택한 카테고리가 있으면 변수+=1을 해준 후, 변수=1이면 작업을 반복하지 않도록 함

<br>

## 5. Results
  ▶ 분석 결과: 신간도서가 트렌드를 반영함을 확인 할 수 있었습니다. 단, 변화의 정도는 카테고리별로 상이했습니다. 
  ▶ 아쉬웠던 점
        - 영어에 비해 형태소 분석에 한계가 있었습니다.
        - '메타 버스'같은 단어는 형태소 분석시 '메타' '버스'로 분리되는 문제를 효과적으로 해결하지는 못했습니다.
        
  



