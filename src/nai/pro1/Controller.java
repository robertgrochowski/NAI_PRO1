package nai.pro1;

import Jama.Matrix;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;

import java.util.List;

public class Controller {

    private boolean[][] selectedFields;
    private NeuralNetwork network;
    private boolean learned = false;

    @FXML
    TextField epochAmount;
    @FXML
    TextField errorThreshold;
    @FXML
    TextField alphaRate;
    @FXML
    Label modelStatus;
    @FXML
    Label predict;
    @FXML
    LineChart<Integer, Double> chart;
    public Controller(){
        selectedFields = new boolean[6][4];
    }

    @FXML
    private void classifyClicked(ActionEvent event){

        if(!learned) {
            alert("Model hasn't been learned yet!");
            return;
        }

        Matrix input = new Matrix(1, 24);
        int p = 0;
        for(int i = 0; i < 6; i++)
        {
            for(int x = 0; x < 4; x++)
                input.set(0, p++, (selectedFields[i][x] ? 1d : 0d ));
        }

        input.print(0, 0);
        int classified = network.classify(input);
        if(classified != -1)
            predict.setText(classified+"");
        else
            predict.setText("?");
    }
    @FXML
    private void learnNetworkClicked(ActionEvent event)
    {
        try {
            double alpha = Double.parseDouble(alphaRate.getText());
            double maxError = Double.parseDouble(errorThreshold.getText());
            int epoch = Integer.parseInt(epochAmount.getText());

            network = new NeuralNetwork(3, alpha);
            List<Double> errors = network.teach(epoch, maxError);
            modelStatus.setText("taught");
            modelStatus.setTextFill(Color.FORESTGREEN);

            XYChart.Series<Integer, Double> series = new XYChart.Series<>();
            for (int i = 0; i < errors.size(); i++) {
                series.getData().add(new XYChart.Data<>(i, errors.get(i)));
            }
            chart.getData().clear();
            chart.getData().add(series);
            learned = true;
        }
        catch(NumberFormatException e){
            alert("Bad number format");
        }
        catch(Exception e) {
            e.printStackTrace();
            alert(e.toString());
        }
    }

    //Code below is responsible for number drawing
    @FXML
    private void mouseEntered(MouseEvent e) {
        Node source = (Node)e.getSource() ;
        Integer colIndex = GridPane.getColumnIndex(source);
        Integer rowIndex = GridPane.getRowIndex(source);

        if(source instanceof Rectangle)
            ((Rectangle) source).setFill(Color.ROYALBLUE);
    }

    @FXML
    private void mouseClicked(MouseEvent e) {
        Node source = (Node)e.getSource() ;
        Integer colIndex = GridPane.getColumnIndex(source);
        Integer rowIndex = GridPane.getRowIndex(source);

        if(source instanceof Rectangle)
        {
            selectedFields[rowIndex][colIndex] = !selectedFields[rowIndex][colIndex];

            if(selectedFields[rowIndex][colIndex])
                ((Rectangle) source).setFill(Color.FORESTGREEN);
            else
                ((Rectangle) source).setFill(Color.TRANSPARENT);
        }
    }

    @FXML
    private void mouseExit(MouseEvent e) {
        Node source = (Node)e.getSource() ;
        Integer colIndex = GridPane.getColumnIndex(source);
        Integer rowIndex = GridPane.getRowIndex(source);

        if(source instanceof Rectangle)
        {
            if(selectedFields[rowIndex][colIndex])
                ((Rectangle) source).setFill(Color.FORESTGREEN);
            else
                ((Rectangle) source).setFill(Color.TRANSPARENT);
        }
    }

    private void alert(String msg) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setContentText(msg);
        alert.show();
    }
}
