# Notebooks

이 프로젝트에서는 전처리 실행과 결과 확인에만 notebook을 사용한다.

```text
notebooks/
├── README.md
└── preprocessing/
    └── 01_prepare_bcic2a.ipynb
```

분류와 복원 실험은 중단 후 seed 단위로 재개할 수 있도록 `scripts/`의 명령행
프로그램으로 실행한다. 모델 구현이나 장시간 GPU 학습 코드를 notebook에 중복해서
추가하지 않는다.
