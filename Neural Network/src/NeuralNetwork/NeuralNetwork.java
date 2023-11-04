package NeuralNetwork;

import java.util.ArrayList;

public class NeuralNetwork {
    
    private ArrayList<Layer> layers;
    private int amountOfInputs;

    public NeuralNetwork(int amountOfInputs){
        this.amountOfInputs = amountOfInputs;
        layers = new ArrayList<Layer>();
    }
    
    public void addLayer(int amountOfNeuronsIn, int amountOfNeuronsOut, ActivationFunc activationFunc){
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
    
    public void learn(double learnRate, double[] inputs, double[] expectedOutputs){
        double cost = getCost(inputs, expectedOutputs);
        for(Layer layer : layers){
            double[][] costWeightGradiants = new double[layer.getAmountOfNodesOut()][layer.getAmountOfNodesIn()];
            double[] costBiasGradiants = new double[layer.getAmountOfNodesOut()];

            //get current weights and biases
            double[][] currentWeights = layer.getWeights();
            double[] currentBiases = layer.getBiases();
            
            //calculate gradiants for weights
            for(int i = 0; i < costWeightGradiants.length; i++){
                for(int j = 0; j <costWeightGradiants[0].length; j++){
                    //approximate slope
                    currentWeights[i][j] += 0.00001;
                    costWeightGradiants[i][j] = (getCost(inputs, expectedOutputs) - cost) / 0.00001;
                    currentWeights[i][j] -= 0.00001;
                }
            }

            //calculate gradiants for biases
            for(int i = 0; i < costBiasGradiants.length; i++){
                //approximate slope
                currentBiases[i] += 0.00001;
                costBiasGradiants[i] = (getCost(inputs, expectedOutputs) - cost) / 0.00001;
                currentBiases[i] -= 0.00001;
            }

            //apply gradiants on layer
            layer.useGradiants(learnRate, costWeightGradiants, costBiasGradiants);
        }


    }

    public double getCost(double[] inputs, double[] expectedOutputs){
        //make sure the amount of expectedOutputs entered is correct
        if(inputs.length != expectedOutputs.length)
            throw new RuntimeException("the given amount of expected outputs is wrong");
        
        double[] outputs = getOutputs(inputs);
        double cost = 0;

        //sum node costs
        for(int i = 0; i < outputs.length; i++){
            cost += Layer.nodeCost(outputs[i], expectedOutputs[i]);
        }

        //return the avg cost
        return cost / outputs.length;
    }

}
