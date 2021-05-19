package edu.upc.svmtraining.util;

import java.io.Serializable;

public class TfIdfFeatureWeighterUtil implements Serializable {
    int numDocs;//数据集文本数目
    int[] df;//包含指定特征值的文本数目,int数组存储的是特征值的表示id

    public TfIdfFeatureWeighterUtil(int numDocs, int[] df) {
        this.numDocs = numDocs;
        this.df = df;
    }

    public double weight(int feature, int frequency) {
        return Math.log10(frequency + 1) * Math.log10((double) this.numDocs / (double) this.df[feature] + 1.0D);
    }
}
