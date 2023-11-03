
public class Layer {
    
    private double[][] weights;
    private double[] biases;

    public Layer(int amountOfNeuronsIn, int amountOfNeuronsOut){
        weights = new double[amountOfNeuronsOut][amountOfNeuronsIn];
        biases = new double[amountOfNeuronsOut];
    }

    public double[] getOutputs(double[] neuronsInput, ActivationFunc activationFunc){
        int amountOfNeuronsIn = weights.length;
        int amountOfNeuronsOut = weights[0].length;

        //check if the length of neuronsInput is correct
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

}
