# 3. First model

A concrete model extends `PHP2xAI\Models\Model` and implements at least `output()` and `loss()`.

```php
use PHP2xAI\Models\Model;
use PHP2xAI\Tensor\Tensor;

class BinaryModel extends Model
{
    public function __construct(?Optimizer $optimizer = null)
    {
        $this->W = Tensor::init([2, 1]);
        $this->b = Tensor::zeros([1]);
        parent::__construct($optimizer);
    }

    public function forward(Tensor $x): Tensor
    {
        return $x->matmul($this->W)->add($this->b);
    }

    public function output(Tensor $x): Tensor
    {
        return $this->forward($x)->sig();
    }

    public function loss(Tensor $x, Tensor $y): Tensor
    {
        return $this->forward($x)->sig()->CE($y)->mean();
    }
}
```

Properties assigned through `$this->W` and `$this->b` become model parameters. The graph registers them as `param`; only trainable parameters are passed to the optimizer.

A typical application flow is:

```php
$train = new StreamFileDataset('./train.txt', 32);
$val = new StreamFileDataset('./val.txt', 32);
$dataset = new TrainValidateDataset($train, $val);

$optimizer = new Adam(0.001);
$model = new BinaryModel($optimizer);
$model->setRuntime('PHP');
$model->train($dataset, 10, './weights.json', 10);
```

To change runtime:

```php
$model->setRuntime('CPP');
```

The model generates the graph and configuration required by the C++ runtime before starting training.
