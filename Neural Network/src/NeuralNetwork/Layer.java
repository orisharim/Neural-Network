package NeuralNetwork;

public class Layer {
    
    private double[][] weights;
    private double[] biases;
    private ActivationFunc activationFunc;

    public Layer(int amountOfNeuronsIn, int amountOfNeuronsOut, ActivationFunc activationFunc){
        weights = new double[amountOfNeuronsOut][amountOfNeuronsIn];
        biases = new double[amountOfNeuronsOut];

        this.activationFunc = activationFunc;
    }

    public double[] getOutputs(double[] neuronsInput){
        int amountOfNeuronsIn = weights[0].length;
        int amountOfNeuronsOut = weights.length;

        //make sure the length of neuronsInput is correct
        if(neuronsInput.length != weights[0].length)
            throw new RuntimeException("the given amount of neurons is wrong");
        
        double[] outputs = new double[amountOfNeuronsOut];

        for(int i = 0; i < amountOfNeuronsOut; i++){
            double output = 0;
            //calculate output after weights
            for(int j = 0; j < amountOfNeuronsIn; i++){
                output += neuronsInput[j] * weights[i][j];
            }
            //add bias to the output
            output += biases[i];
            //run activation function on output
            output = activationFunc.func(output);
            //save output
            outputs[i] = output;
        }

        return outputs;
    }

    public void useGradiants(double learnRate, double[][] costWeightGradiants, double[] costBiasGradiants){
        //make sure the length of cost bias gradients is correct
        if(costBiasGradiants.length != biases.length)
            throw new RuntimeException("the given amount of cost biases gradiants is wrong");

        //apply cost bias gradiants
        for(int i = 0; i < biases.length; i++){
            biases[i] -= costBiasGradiants[i] * learnRate;
        }
        
        //make sure the length of cost bias gradients is correct
        if(weights.length != costWeightGradiants.length || weights[0].length != costWeightGradiants[0].length)
            throw new RuntimeException("the given amount of cost weight gradiants is wrong");


        //apply cost weight gradiants
        for(int i = 0; i < weights.length; i++){
            for(int j = 0; j < weights[0].length; j++){
                weights[i][j] -= costWeightGradiants[i][j] * learnRate;
            }
        }
    }

    public static double nodeCost(double outputValue, double expectedOutputValue){
        double error = outputValue - expectedOutputValue;
        return error * error;
    }

    public static double nodeCostDerivative(double outputValue, double expectedOutputValue){
        return (outputValue - expectedOutputValue) * 2;
    }

    public double[][] getWeights(){
        return weights;
    }

    public double[] getBiases(){
        return biases;
    }

    public int getAmountOfNodesIn(){
        return weights[0].length;
    }

    public int getAmountOfNodesOut(){
        return weights.length;
    }

}
