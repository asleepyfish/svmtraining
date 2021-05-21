package edu.upc.svmtraining.console;

import edu.upc.svmtraining.classifier.SVMClassifier;
import edu.upc.svmtraining.model.SVMModel;
import edu.upc.svmtraining.util.FileOperationUtil;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;

import static com.hankcs.hanlp.utility.Predefine.logger;

public class SVMClassifierConsole {
    //训练集路径
    private static final String CORPUS_FOLDER = "data/搜狗文本分类语料库微型版/训练集";
    //测试集路径
    private static final String TEST_SET = "data/搜狗文本分类语料库微型版/测试集";
    //命中分类数
    private static int hit_num = 0;
    //测试分类数
    private static int test_num = 0;
    private static int sport_predict_num = 0, healthy_predict_num = 0, military_predict_num = 0, education_predict_num = 0, travel_predict_num = 0, car_predict_num = 0, economics_predict_num = 0;
    private static int sport_test_num = 0, healthy_test_num = 0, military_test_num = 0, education_test_num = 0, travel_test_num = 0, car_test_num = 0, economics_test_num = 0;
    private static int sport_hit_num = 0, healthy_hit_num = 0, military_hit_num = 0, education_hit_num = 0, travel_hit_num = 0, car_hit_num = 0, economics_hit_num = 0;
    //模型保存路径
    public static final String MODEL_PATH = "data/svm-classification-model.ser";

    public static void main(String[] args) throws IOException {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String startTime = sdf.format(new Date());
        long startTimeMills = System.currentTimeMillis();
        System.out.println("测试程序开始时间：" + startTime);
        System.out.println("=============================");
        SVMClassifier classifier = new SVMClassifier(trainOrLoadModel());
        FileOperationUtil.traverseFile(TEST_SET);
        Map<String, String> map = FileOperationUtil.map;

        for (Map.Entry<String, String> entry : map.entrySet()) {
            String path = entry.getKey();
            String category = entry.getValue();
            getCategoryNum(path, category, classifier);
        }

        printTestAccuracy("体育", sport_hit_num, sport_test_num, sport_predict_num);
        printTestAccuracy("健康", healthy_hit_num, healthy_test_num, healthy_predict_num);
        printTestAccuracy("军事", military_hit_num, military_test_num, military_predict_num);
        printTestAccuracy("教育", education_hit_num, education_test_num, education_predict_num);
        printTestAccuracy("旅游", travel_hit_num, travel_test_num, travel_predict_num);
        printTestAccuracy("汽车", car_hit_num, car_test_num, car_predict_num);
        printTestAccuracy("财经", economics_hit_num, economics_test_num, economics_predict_num);

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
     * @param path       路径
     * @param category   分类名
     * @param classifier 分类器对象
     * @return 返回是否分类正确
     * @throws IOException Exception
     */
    private static boolean accuracyOfTest(String path, String category, SVMClassifier classifier) throws IOException {
        String readFile = FileOperationUtil.readFile(path);
        String predict = predict(classifier, readFile);
        switch (predict) {
            case "体育":
                sport_predict_num++;
                break;
            case "健康":
                healthy_predict_num++;
                break;
            case "军事":
                military_predict_num++;
                break;
            case "教育":
                education_predict_num++;
                break;
            case "旅游":
                travel_predict_num++;
                break;
            case "汽车":
                car_predict_num++;
                break;
            case "财经":
                economics_predict_num++;
                break;
            default:
                break;
        }
        return predict.equals(category);
    }

    /**
     * 预测分类
     *
     * @param classifier 分类器对象
     * @param text       分类文本
     * @return 分类结果
     */
    private static String predict(SVMClassifier classifier, String text) {
        /*
          LinearSVMClassifier是最顶层的接口
          调用AbstractClassifier的classify方法（多态）,将文本作为参数预测去对比分类,
          对于不同分类结果和预期特征值用Map<String,Double>来存
          最终返回的是经过特征向量比对后此map中得分最高的分类
         */
        return classifier.classify(text);
    }

    /**
     * 训练模型
     *
     * @return 分类模型
     * @throws IOException Exception
     */
    private static SVMModel trainOrLoadModel() throws IOException {
        SVMModel model = (SVMModel) readObjectFrom(MODEL_PATH);
        if (model != null) {
            return model;
        }
        // 创建分类器
        SVMClassifier classifier = new SVMClassifier();
        // 训练后的模型支持持久化,文件编码类型UTF-8
        classifier.train(CORPUS_FOLDER, "UTF-8");
        model = (SVMModel) classifier.getModel();
        saveObjectTo(model, MODEL_PATH);
        return model;
    }

    /**
     * 序列化对象
     *
     * @param o    存储对象
     * @param path 存储路径
     */
    public static void saveObjectTo(Object o, String path) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path));
            oos.writeObject(o);
            oos.close();
        } catch (IOException e) {
            logger.warning("在保存对象" + o + "到" + path + "时发生异常" + e);
        }

    }

    /**
     * 反序列化对象
     *
     * @param path 读取路径
     * @return 读取模型
     */
    public static Object readObjectFrom(String path) {
        ObjectInputStream ois;
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
     * @param path       路径
     * @param category   分类名
     * @param classifier 分类器对象
     * @throws IOException Exception
     */
    public static void getCategoryNum(String path, String category, SVMClassifier classifier) throws IOException {
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
     * @param category 分类名
     * @param hit_num  命中数
     * @param test_num 测试数
     */
    public static void printTestAccuracy(String category, int hit_num, int test_num, int predict_num) {
        double precision = hit_num / (test_num * 1.0);
        double recall = hit_num / (predict_num * 1.0);
        System.out.println(category + "类测试集共" + test_num + "个");
        System.out.println("实际为" + category + "类文本共" + hit_num + "个");
        System.out.println("判断为属于" + category + "类文本共" + predict_num + "个");
        System.out.println(category + "类准确率" + precision * 100 + "%");
        System.out.print(category + "类召回率");
        System.out.printf("%.2f", recall * 100);
        System.out.println("%");
        System.out.print(category + "类F1值");
        System.out.printf("%.2f", ((2 * precision * recall) / (precision + recall)) * 100);
        System.out.println("%");
        System.out.println("=============================");
    }
}
