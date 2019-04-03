package nai.pro1;

import Jama.Matrix;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private int neuronAmount;
    private Matrix weights;

    private double alpha;
    private Matrix bias;

    public NeuralNetwork(int neuronAmount, double alpha)
    {
        this.neuronAmount = neuronAmount;
        this.weights = Matrix.random(this.neuronAmount, 24); //TODO
        this.alpha = alpha;

        bias = new Matrix(neuronAmount, 1, 1d); //TODO
    }

    //Returns error values
    public List<Double> teach(Matrix trainingSet, Matrix answerSet, int epochAmount, double maxError) {

        List<Double> errorList = new ArrayList<>();

        for(int i = 0; i < epochAmount; i++) {

            for(int s = 0; s < trainingSet.getRowDimension(); s++)
            {

                Matrix thisInput = trainingSet.getMatrix(s, s, 0, trainingSet.getColumnDimension() - 1).transpose();
                Matrix thisAnswer = answerSet.getMatrix(s, s, 0, answerSet.getColumnDimension() - 1).transpose();
                Matrix NETValue = weights.times(thisInput).plus(bias);

                Matrix output = getFunctionValue(NETValue);

                errorList.add(getError(output, thisAnswer));
                System.out.println("error="+errorList.get(errorList.size()-1));
                //improve
                weights = thisAnswer.minus(output).
                        times(thisInput.transpose()).
                        times(alpha).
                        plus(weights);

                bias = thisAnswer.minus(output).times(alpha).plus(bias);

                //thisAnswer.minus(output).print(0, 3);
            }
        }
        return errorList;
    }

    public int classify(Matrix input)
    {
        Matrix NETValue = weights.times(input.transpose()).plus(bias);
        Matrix output = getFunctionValue(NETValue);
        output.print(1,0);

        int sum = 0;
        for(int i = 0; i < output.getRowDimension(); i++)
            sum += output.get(i, 0);

        if(sum != 1)
            return -1;

        for(int i = 0; i < output.getRowDimension(); i++)
            if(output.get(i, 0) == 1)
                return i;
        return -1;
    }


    double getError(Matrix outputs, Matrix answerSet) {
        double error = 0;
        Matrix diff = answerSet.minus(outputs);

        for (int row = 0; row < diff.getRowDimension(); row++) {
            for(int col = 0; col < diff.getColumnDimension(); col++)
            {
                error += Math.abs(diff.get(row, col)); //instead of square
            }
        }
        return error * 0.5;
    }

   Matrix getFunctionValue(Matrix NETValue) {
        Matrix output = new Matrix(NETValue.getRowDimension(), 1);
        for(int i = 0; i < NETValue.getRowDimension(); i++)
        {
            if(NETValue.get(i, 0 ) >= 0)
                output.set(i, 0, 1d);
            else output.set(i, 0, 0d);
        }
        return output;
   }

}
