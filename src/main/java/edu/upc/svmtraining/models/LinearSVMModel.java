package edu.upc.svmtraining.models;

import com.hankcs.hanlp.classification.features.IFeatureWeighter;
import com.hankcs.hanlp.classification.models.AbstractModel;
import de.bwaldvogel.liblinear.Model;

/**
 * 线性SVM模型
 */
public class LinearSVMModel extends AbstractModel {
    /**
     * 训练样本数
     */
    public int n = 0;
    /**
     * 类别数
     */
    public int c = 0;
    /**
     * 特征数
     */
    public int d = 0;
    /**
     * 特征权重计算工具
     */
    public IFeatureWeighter featureWeighter;
    /**
     * SVM分类模型
     */
    public Model svmModel;
}