# Google hard-case 이미지 수집 설계

> 대상 독자: Claude 에이전트 또는 구현 에이전트  
> 목적: `gecko-vision-gate` v0 detector의 운영 성능을 보강하기 위해 구글 이미지와 운영 프레임을 수집, 정리, 라벨링하는 설계

## 1. 한 줄 정의

구글 이미지는 일반 게코 사진을 늘리는 용도가 아니다. 운영 카메라에서 detector가 놓치기 쉬운 **야간 IR/흑백**과 **가림/부분 노출** 장면을 채우는 hard-case 보강 데이터다.

## 2. 배경

현재 `datasets/raw/crawl_breeder/`에는 분양샵/브리더 이미지가 7,830장 있다. 이 데이터는 밝은 주간, 클로즈업, 선명한 개체 사진에 치우쳐 있다.

운영 환경은 다르다.

```text
운영 환경:
- 야간 IR 또는 흑백
- 원거리 펫캠 시점
- 코르크, 은신처, 나무, 식물, 물그릇에 가림
- 머리, 몸통, 꼬리 일부만 보임
- 배경과 게코 색이 섞임
```

v0의 핵심 실패는 false negative다.

```text
false negative = 게코가 있는데 detector가 없다고 판단
결과 = Claude RBA Worker가 분석해야 할 클립을 skip할 위험
```

따라서 이번 수집은 예쁜 사진 수집이 아니라 false negative를 줄이는 데이터 보강 작업이다.

## 3. v0 목표

총량은 800~1,200장으로 시작한다.

```text
총 목표: 800~1,200장

gecko present bbox 대상: 600~800장
negative/no-gecko: 200~300장
ambiguous/occluded review: 100장 내외
```

도메인 비중은 아래를 목표로 한다.

```text
ir_night: 35%
occluded: 25%
ir_night_occluded: 25%
normal/day/other: 15%
```

`ir_night_occluded`가 최우선이다. 야간이면서 일부만 보이는 장면은 운영 false negative와 가장 가깝다.

## 4. 수집 원칙

### 4.1 운영 기준 우선

운영 프레임이 일부 있으므로, 먼저 운영 프레임을 기준점으로 삼는다.

```text
1. 운영 영상에서 야간/가림 프레임을 100~300장 추출
2. 추출 프레임을 보고 부족한 장면 유형을 정의
3. 구글 이미지로 부족분을 채움
```

구글 이미지는 운영 프레임을 대체하지 않는다. 운영에서 부족한 hard case를 보완한다.

### 4.2 test split은 운영 프레임만 사용

크롤링 이미지는 test split에 넣지 않는다.

```text
test = operational only
val = operational 중심, 필요 시 crawl_google 소량
train = operational + crawl_google + crawl_nighttime + negative
```

이유는 운영 성능을 보려면 평가 데이터가 운영 도메인이어야 하기 때문이다. 구글 이미지로 평가하면 실제 펫캠 성능이 가려진다.

### 4.3 밝은 클로즈업은 낮은 우선순위

`crawl_breeder`가 이미 밝은 주간 클로즈업을 충분히 갖고 있다. 구글 수집에서는 다음 이미지를 낮은 우선순위로 둔다.

```text
낮은 우선순위:
- 선명한 스튜디오 사진
- 정면 클로즈업
- 흰 배경 상품 이미지
- 게코 전체가 크게 보이는 주간 사진
```

## 5. 저장 위치

기존 데이터셋 구조를 그대로 따른다.

```text
datasets/raw/
├── operational/<clip_id>/       # 운영 영상 프레임
├── crawl_google/                # 일반 구글 hard-case 이미지
├── crawl_nighttime/             # 야간/IR/흑백 중심 이미지
└── negative/                    # 게코 없는 운영 프레임 또는 유사 배경
```

권장 분류:

```text
운영 영상 프레임:
  datasets/raw/operational/<clip_id>/

구글 야간/IR/흑백:
  datasets/raw/crawl_nighttime/

구글 가림/부분 노출:
  datasets/raw/crawl_google/

게코 없는 운영 프레임:
  datasets/raw/negative/
```

## 6. manifest 정책

현재 `datasets/manifest.csv` 컬럼은 아래와 같다.

```text
filename,source,clip_id,split,labeled,domain
```

v0에서는 이 스키마를 깨지 않는다. 필요한 메타데이터는 우선 별도 파일로 둔다.

```text
datasets/source_metadata.csv
```

권장 컬럼:

```text
filename
search_query
source_url
page_url
license_hint
downloaded_at
collector
notes
```

이유:
- `scripts/build_manifest.py`는 `split`, `labeled`, `domain`만 보존한다.
- manifest에 새 컬럼을 즉석으로 추가하면 기존 스크립트 수정이 필요하다.
- v0 문서화 목적에는 별도 metadata CSV가 더 안전하다.

## 7. domain 태그

`manifest.csv`의 `domain` 값은 v0에서 다음 값만 사용한다.

```text
ir_night
occluded
ir_night_occluded
negative
day_other
```

판정 기준:

```text
ir_night:
  흑백, IR, 야간 조명, 저조도 이미지. 게코가 대부분 보일 수 있음.

occluded:
  주간/컬러여도 은신처, 코르크, 나무, 식물, 물그릇 등에 일부가 가려짐.

ir_night_occluded:
  야간/흑백 조건이면서 일부만 보이거나 구조물 뒤에 숨음.

negative:
  게코가 없음. 은신처, 바닥재, 조명 변화, 그림자, 물그릇, 나무 등 혼동 요소 포함.

day_other:
  위 hard case가 아니지만 다양성 유지용으로 남기는 이미지.
```

도메인이 애매하면 `ir_night_occluded`를 남발하지 않는다. 리뷰 큐에 두고 `notes`에 이유를 적는다.

## 8. 구글 검색어 전략

검색어는 영어 중심으로 시작한다. 한국어 검색어는 보조로 사용한다.

### 8.1 야간 IR/흑백

```text
crested gecko night vision
crested gecko infrared camera
crested gecko night camera
crested gecko black and white camera
gecko night vision camera
gecko infrared night
gecko security camera night
leopard gecko night vision
```

### 8.2 가림/부분 노출

```text
crested gecko hiding
crested gecko hide
crested gecko behind cork bark
crested gecko in enclosure hiding
crested gecko partially visible
gecko hiding behind branch
gecko hiding in plants
gecko in cork bark
gecko peeking out hide
```

### 8.3 야간 + 가림 조합

```text
crested gecko night hiding
crested gecko night enclosure
crested gecko night camera hiding
gecko night vision enclosure
gecko infrared hiding
gecko hiding at night
```

### 8.4 negative 후보

구글 negative는 낮은 우선순위다. 가능하면 운영 프레임에서 뽑는다.

```text
reptile enclosure night vision no animal
crested gecko enclosure night empty
terrarium night vision empty
gecko enclosure hide cork bark no gecko
```

## 9. 수집 방식

### 9.1 v0 권장 방식

v0는 반자동 수집을 권장한다.

```text
1. 검색어 큐를 만든다.
2. 검색 결과에서 hard case만 사람이 선별한다.
3. 이미지를 raw 폴더에 저장한다.
4. source_metadata.csv에 출처와 검색어를 기록한다.
5. scripts/build_manifest.py를 실행한다.
6. manifest에서 domain, split, labeled를 채운다.
```

완전 자동 다운로드는 v0 기본이 아니다. 검색 결과에는 잘못된 동물, 그림, 상품 이미지, 중복 이미지, 저작권 위험 이미지가 섞인다.

### 9.2 API 사용 시 주의

Google Programmable Search JSON API는 이미지 검색 결과를 JSON으로 받을 수 있고, `searchType=image`, `rights` 같은 파라미터를 지원한다. 다만 2026년 현재 신규 고객 제한과 전환 일정 이슈가 있으므로 사용 가능 여부를 먼저 확인해야 한다.

API가 가능하면 다음을 지킨다.

```text
필수 기록:
- query
- image URL
- containing page URL
- license/right hint
- download date
- original file extension
```

API가 불가능하면 브라우저 수동 선별 + metadata CSV 기록으로 진행한다.

## 10. 중복 제거

v0에서는 두 단계로 한다.

```text
1단계: 파일 해시 중복 제거
  같은 파일 바이트가 반복 저장된 경우 제거

2단계: 육안 중복 제거
  같은 사진의 리사이즈, 크롭, 워터마크 차이는 리뷰에서 제거
```

perceptual hash는 v1 backlog로 둔다. v0 완료 조건에는 포함하지 않는다.

## 11. 라벨링 정책

학습은 object detection이므로 gecko present 이미지에는 bbox가 필요하다.

```text
class:
- gecko
```

라벨링 기준:

```text
게코 전체가 보임:
  보이는 전체 몸체를 bbox로 감싼다.

일부만 보임:
  실제로 보이는 부분만 bbox로 감싼다.

가림이 심함:
  보이는 부분이 게코라고 확신 가능하면 bbox.
  확신이 어렵다면 ambiguous 리뷰 큐.

게코 없음:
  annotation 없음. negative 이미지로 둔다.
```

야간/IR 이미지는 contrast가 낮더라도 사람이 게코를 식별할 수 있으면 라벨링 대상이다.

## 12. split 정책

v0 split 권장:

```text
train: 70%
val: 15%
test: 15%
```

단, test는 반드시 운영 프레임만 사용한다.

```text
train:
  operational + crawl_google + crawl_nighttime + negative

val:
  operational 중심
  crawl_google/crawl_nighttime은 소량만 허용

test:
  operational only
  clip_id 단위로 분리
```

운영 프레임은 clip 단위로 split한다. 같은 영상에서 나온 프레임이 train과 test에 동시에 들어가면 안 된다.

## 13. 품질 게이트

수집 완료 기준:

```text
총 이미지: 800~1,200장
gecko present bbox 후보: 600장 이상
negative: 200장 이상
ir_night + ir_night_occluded: 전체의 50% 이상
source_metadata.csv 기록률: 95% 이상
test split: 100% operational
```

라벨링 완료 기준:

```text
present 이미지 bbox 라벨 완료
negative 이미지 annotation 없음 확인
ambiguous 이미지는 train에서 제외하거나 별도 리뷰
domain 태그 누락 없음
```

성능 평가 기준:

```text
1순위: operational test false negative 감소
2순위: ir_night recall
3순위: ir_night_occluded recall
4순위: false positive는 negative set에서 확인
```

v0에서는 precision보다 recall을 우선한다. Claude 분석 전에 버리는 위험이 더 크기 때문이다.

## 14. 에이전트 작업 순서

Claude 에이전트가 이 문서를 읽고 작업할 때는 아래 순서를 따른다.

```text
1. datasets/README.md 확인
2. docs/MODEL_AND_TRAINING_PLAN.md 확인
3. 운영 프레임 보유량 확인
4. 검색어 큐 작성
5. source_metadata.csv 스키마 초안 작성
6. 수동 또는 반자동 수집 SOP 작성
7. build_manifest.py와 충돌 없는지 확인
8. domain/split/labeled 입력 방식 정리
9. 라벨링 툴 또는 COCO 변환 계획 작성
10. test split이 operational only인지 검증하는 체크 추가
```

## 15. 이번 설계에서 하지 않는 것

```text
하지 않음:
- YOLO 또는 RF-DETR 학습 실행
- 자동 대량 다운로드를 기본값으로 설정
- manifest.csv 스키마 즉석 변경
- test split에 구글 이미지 추가
- 라이선스가 불명확한 이미지를 상용 데이터셋으로 확정
```

상용 또는 공개 배포로 넘어가면 이미지 라이선스와 출처 정책을 다시 검토해야 한다.

## 16. 구현 메모

현재 `scripts/build_manifest.py`는 raw 이미지 스캔 후 `filename`, `source`, `clip_id`, `split`, `labeled`, `domain`만 쓴다. 그러므로 구현 에이전트는 아래 중 하나를 선택해야 한다.

```text
보수적 선택:
  source_metadata.csv를 별도 관리한다.

확장 선택:
  manifest.csv 컬럼을 확장하고 build_manifest.py의 preserve 컬럼을 함께 수정한다.
```

v0는 보수적 선택을 권장한다. 데이터 스키마를 바꾸지 않고도 수집과 라벨링을 시작할 수 있다.
