* `field_dims: list`  
    각 원소는 각 category(Action, Adventure, ..., age, gender)의 개수를 나타냄 

### FM + original
* `CustomDataset -> cats, conts, target, c`
* class `FM`
    * `forward(self, x, x_cont, emb_x)`  
        `return x, cont_emb, lin_term, inter_term`
    * `training_step(self, batch, batch_idx)`  
        `x, x_cont, y, c_values = batch`  
        `return loss_y`

### original/layers
##### FeatureEmbedding
* `field_dims`  
`np.array([943, 1650, 2, 2, 2, ..., 2, 10, 2, 21])`
* `offsets` ; len:24  
`np.array([0  943 2593 2595 2597 ..., 2629, 2631, 2641, 2643])`
* `x[0]` ; x.shape = [4096, 24]  
`[ 894 1248  0    ...  4    0   10]` @ line:33  
`[ 894 2191 2593 ... 2635 2641 2653]` @ line:34  
`self.embedding(x)`를 이용해 `int`값마다 16차원의 벡터 부여  
ex) `2593 = tensor([ 0.7755, -1.3483, ... 1.8129,  0.1935])`

##### FM_Linear
* `linear_term`
