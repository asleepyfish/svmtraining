package edu.upc.svmtraining;

import edu.upc.svmtraining.classifier.SVMClassifier;

public class Test {
    public static void main(String[] args) {
        SVMClassifier sc = new SVMClassifier();
        sc.predict("货币,货币,电子");
    }
}
