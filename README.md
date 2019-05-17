# magnet-loss

An encoder for learning embeddings via minimizing the magnet loss. In the embedding space each class forms its own cluster. See "METRIC LEARNING WITH ADAPTIVE DENSITY DISCRIMINATION" by Rippel et al. The codes from the following repos are used:
* https://github.com/pumpikano/tf-magnet-loss
* https://github.com/nwojke/cosine_metric_learning/blob/master/losses.py

Batch Losses
```
plot_smooth(batch_losses)
```
![batch_losses](https://raw.githubusercontent.com/youngklee/magnet-loss/master/batch_losses.png)

Initial Embeddings of the MNIST dataset
```
plot_embedding(initial_reps[:500], y_train[:500])
```
![initial_embeddings](https://raw.githubusercontent.com/youngklee/magnet-loss/master/initial_embeddings.png)

Final Embeddings
```
plot_embedding(final_reps[:500], y_train[:500])
```
![final_embeddings](https://raw.githubusercontent.com/youngklee/magnet-loss/master/final_embeddings.png)