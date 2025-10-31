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
