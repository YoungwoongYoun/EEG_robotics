# Local dataset layout

BCI Competition IV Dataset 2a 파일은 저장소에 commit하지 않고 다음 위치에 둔다.

```text
data/
├── raw/
│   ├── A01T.gdf
│   ├── A01E.gdf
│   └── ... A09T.gdf, A09E.gdf
└── labels/
    ├── A01E.mat
    └── ... A09E.mat
```

원본 GDF는 BCI Competition IV Dataset 2a 배포본을 사용하고, evaluation label은
competition results 페이지의 공식 true-label 파일을 사용한다.
