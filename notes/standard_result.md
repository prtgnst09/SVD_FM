|   |**original+FM** |**original+DeepFM** |**SVD+FM** |**SVD+DeepFM** |
|---|---             |---                 |---        |---            |
|precision |0.450000000000000|0.456880733944954 |0.198165137614678 |0.5279816513761467 |
|3         |      ``         |         ``       |0.202752293577981 |0.5279816513761468 |
---
<a name='footnote_1'>1</a> : `original`에선 `weight_decay` , `SVD`에서는 `L2-norm` 사용  & 모든 option에 대해 `isuniform`을 `False`로 사용  
<a name='footnote_2'>3</a> : 전부 `weight_decay`를 사용  