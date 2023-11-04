package NeuralNetwork;

import java.util.ArrayList;

public class NeuralNetwork {
    
    private ArrayList<Layer> layers;
    private int amountOfInputs;
    
    public NeuralNetwork(int amountOfInputs){
        this.amountOfInputs = amountOfInputs;
        layers = new ArrayList<Layer>();
    }
    
    public void addHiddenLayer(int amountOfNeuronsIn, int amountOfNeuronsOut, ActivationFunc activationFunc){
        layers.add(new Layer(amountOfNeuronsIn, amountOfNeuronsOut, activationFunc));
    }

    public double[] getOutputs(double[] inputs){
        //make sure the amount of inputs entered is correct
        if(inputs.length != amountOfInputs)
            throw new RuntimeException("the given amount of inputs is wrong");

        //go through all outputs 
        for(Layer layer : layers) {
            inputs = layer.getOutputs(inputs);
        }
        return inputs;
    }


}
