# magnet-loss

Use an encoder to learn embeddings using the magnet loss. See "METRIC LEARNING WITH ADAPTIVE DENSITY DISCRIMINATION" by Rippel et al.

```
$ plot_smooth(batch_losses)
```
![batch_losses](https://raw.githubusercontent.com/youngklee/magnet-loss/master/batch_losses.png)

```
$ plot_embedding(initial_reps[:400], y_train[:400])
```
![initial_embeddings](https://raw.githubusercontent.com/youngklee/magnet-loss/master/initial_embeddings.png)

```
$ plot_embedding(final_reps[:400], y_train[:400])
```
![final_embeddings](https://raw.githubusercontent.com/youngklee/magnet-loss/master/final_embeddings.png)