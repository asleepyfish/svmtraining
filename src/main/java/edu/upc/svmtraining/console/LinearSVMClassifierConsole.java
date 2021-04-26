package edu.upc.svmtraining.console;

import com.hankcs.hanlp.classification.classifiers.IClassifier;
import edu.upc.svmtraining.classifiers.LinearSVMClassifier;
import edu.upc.svmtraining.models.LinearSVMModel;
import edu.upc.svmtraining.util.FileOperationUtil;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

public class LinearSVMClassifierConsole {
    public static final String CORPUS_FOLDER = "data/搜狗文本分类语料库微型版/训练集";
    public static final String TEST_SET = "data/搜狗文本分类语料库微型版/测试集";
    public static int hit_num = 0;
    public static int test_num = 0;
    public static int sport_test_num = 0, healthy_test_num = 0, military_test_num = 0, education_test_num = 0, travel_test_num = 0, car_test_num = 0, economics_test_num = 0;
    public static int sport_hit_num = 0, healthy_hit_num = 0, military_hit_num = 0, education_hit_num = 0, travel_hit_num = 0, car_hit_num = 0, economics_hit_num = 0;
    /**
     * 模型保存路径
     */
    public static final String MODEL_PATH = "data/svm-classification-model.ser";

    public static void main(String[] args) throws IOException {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String startTime = sdf.format(new Date());
        long startTimeMills = System.currentTimeMillis();
        System.out.println("测试程序开始时间：" + startTime);
        System.out.println("=============================");
        IClassifier classifier = new LinearSVMClassifier(trainOrLoadModel());
        FileOperationUtil.traverseFile(TEST_SET);
        Map<String, String> map = FileOperationUtil.map;
        Set<Map.Entry<String, String>> set = map.entrySet();
        Iterator<Map.Entry<String, String>> it = set.iterator();
        while (it.hasNext()) {
            Map.Entry<String, String> entry = it.next();
            String path = entry.getKey();
            String category = entry.getValue();
            getCategoryNum(path, category, classifier);
        }

        printTestAccuracy("体育", sport_hit_num, sport_test_num);
        printTestAccuracy("健康", healthy_hit_num, healthy_test_num);
        printTestAccuracy("军事", military_hit_num, military_test_num);
        printTestAccuracy("教育", education_hit_num, education_test_num);
        printTestAccuracy("旅游", travel_hit_num, travel_test_num);
        printTestAccuracy("汽车", car_hit_num, car_test_num);
        printTestAccuracy("财经", economics_hit_num, economics_test_num);

        System.out.println("测试集总数共" + test_num + "个");
        System.out.println("测试集中命中的共检测出" + hit_num + "个");
        System.out.println("系统分类的准确率为" + (hit_num / (test_num * 1.0)) * 100 + "%");
        System.out.println("=============================");
        String endTime = sdf.format(new Date());
        long endTimeMills = System.currentTimeMillis();
        System.out.println("测试程序结束时间" + endTime);
        System.out.println("测试程序共耗时" + (endTimeMills - startTimeMills) + "毫秒");


    }

    /**
     * 根据测试集数据测试分类是否准确
     *
     * @param path
     * @param category
     * @param classifier
     * @return
     * @throws IOException
     */
    private static boolean accuracyOfTest(String path, String category, IClassifier classifier) throws IOException {
        String readFile = FileOperationUtil.readFile(path);
        if (predict(classifier, readFile).equals(category)) {
            return true;
        }
        return false;
    }

    /**
     * 预测分类
     *
     * @param classifier
     * @param text
     * @return
     */
    private static String predict(IClassifier classifier, String text) {
        return classifier.classify(text);
    }

    /**
     * 训练模型
     *
     * @return
     * @throws IOException
     */
    private static LinearSVMModel trainOrLoadModel() throws IOException {
        LinearSVMModel model = (LinearSVMModel) readObjectFrom(MODEL_PATH);
        if (model != null) {
            return model;
        }

        File corpusFolder = new File(CORPUS_FOLDER);
        if (!corpusFolder.exists() || !corpusFolder.isDirectory()) {
            System.err.println("没有文本分类语料");
            System.exit(1);
        }
        // 创建分类器
        IClassifier classifier = new LinearSVMClassifier();
        // 训练后的模型支持持久化
        classifier.train(CORPUS_FOLDER);
        model = (LinearSVMModel) classifier.getModel();
        saveObjectTo(model, MODEL_PATH);
        return model;
    }

    /**
     * 序列化对象
     *
     * @param o
     * @param path
     * @return
     */
    public static boolean saveObjectTo(Object o, String path) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path));
            oos.writeObject(o);
            oos.close();
        } catch (IOException e) {
            logger.warning("在保存对象" + o + "到" + path + "时发生异常" + e);
            return false;
        }

        return true;
    }

    /**
     * 反序列化对象
     *
     * @param path
     * @return
     */
    public static Object readObjectFrom(String path) {
        ObjectInputStream ois = null;
        try {
            ois = new ObjectInputStream(new FileInputStream(path));
            Object o = ois.readObject();
            ois.close();
            return o;
        } catch (Exception e) {
            logger.warning("在从" + path + "读取对象时发生异常" + e);
        }

        return null;
    }

    /**
     * 封装分类数量统计的方法
     *
     * @param path
     * @param category
     * @param classifier
     * @throws IOException
     */
    public static void getCategoryNum(String path, String category, IClassifier classifier) throws IOException {
        switch (category) {
            case "体育":
                sport_test_num++;
                test_num++;
                if (accuracyOfTest(path, category, classifier)) {
                    sport_hit_num++;
                    hit_num++;
                }
                break;
            case "健康":
                healthy_test_num++;
                test_num++;
                if (accuracyOfTest(path, category, classifier)) {
                    healthy_hit_num++;
                    hit_num++;
                }
                break;
            case "军事":
                military_test_num++;
                test_num++;
                if (accuracyOfTest(path, category, classifier)) {
                    military_hit_num++;
                    hit_num++;
                }
                break;
            case "教育":
                education_test_num++;
                test_num++;
                if (accuracyOfTest(path, category, classifier)) {
                    education_hit_num++;
                    hit_num++;
                }
                break;
            case "旅游":
                travel_test_num++;
                test_num++;
                if (accuracyOfTest(path, category, classifier)) {
                    travel_hit_num++;
                    hit_num++;
                }
                break;
            case "汽车":
                car_test_num++;
                test_num++;
                if (accuracyOfTest(path, category, classifier)) {
                    car_hit_num++;
                    hit_num++;
                }
                break;
            case "财经":
                economics_test_num++;
                test_num++;
                if (accuracyOfTest(path, category, classifier)) {
                    economics_hit_num++;
                    hit_num++;
                }
                break;
            default:
                break;
        }
    }

    /**
     * 封装打印输出的方法
     *
     * @param category
     * @param hit_num
     * @param test_num
     */
    public static void printTestAccuracy(String category, int hit_num, int test_num) {
        System.out.println(category + "类测试集共" + test_num + "个");
        System.out.println(category + "类测试集中命中的共检测出" + hit_num + "个");
        System.out.println(category + "类测试集准确率" + (hit_num / (test_num * 1.0)) * 100 + "%");
        System.out.println("=============================");
    }
}
