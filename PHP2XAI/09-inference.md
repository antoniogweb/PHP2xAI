# 9. Inference

To load a model:

```php
$model->setRuntime('CPP');
$model->loadModel('./model.json', './weights.json');
```

Then:

```php
$output = $model->predict($x);
$label = $model->predictLabelInt($x);
```

`predict()` returns the graph output; `predictLabelInt()` returns the index of the class with the highest value.

The runtime must be in evaluation mode. This is why `GraphRuntime` defaults to `training = false`. If the graph contains dropout, the operation returns the input unchanged in this mode.

It is nevertheless preferable to build `output()` without dropout and use a separate inference graph.
