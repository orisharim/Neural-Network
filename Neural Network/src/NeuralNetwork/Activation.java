package NeuralNetwork;

public class Activation {
    public static final ActivationFunc SIGMOID = new Sigmoid(); 
    public static final ActivationFunc RELU = new ReLU();
}


interface ActivationFunc{
    public double func(double input);
    public double derivative(double input);
}

class Sigmoid implements ActivationFunc{

    @Override
    public double func(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
        double tmp = func(input);
        return tmp * (1 - tmp);
    }

}

class ReLU implements ActivationFunc{

    @Override
    public double func(double input) {
        return Math.max(0, input);
    }

    @Override
    public double derivative(double input) {
        return (input > 0)? 1 : 0;
    }

}
