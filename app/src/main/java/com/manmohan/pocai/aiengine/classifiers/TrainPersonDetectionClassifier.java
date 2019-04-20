package com.manmohan.pocai.aiengine.classifiers;

import android.content.ContentResolver;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.ParcelFileDescriptor;

import com.manmohan.pocai.aiengine.Classifier;

import java.io.FileDescriptor;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.manmohan.pocai.aiengine.env.FileUtils;
import com.manmohan.pocai.aiengine.wrapper.FaceNet;
import com.manmohan.pocai.aiengine.wrapper.LibSVM;
import com.manmohan.pocai.aiengine.wrapper.MTCNN;

import androidx.core.util.Pair;

import static com.manmohan.pocai.aiengine.env.ImageUtils.getBitmapFromUri;

public class TrainPersonDetectionClassifier {

    public static final int EMBEDDING_SIZE = 512;
    private static TrainPersonDetectionClassifier classifier;
    private static final int FACE_SIZE = 160;


    private MTCNN mtcnn;
    private FaceNet faceNet;
    private LibSVM svm;

    private List<String> classNames;

    private TrainPersonDetectionClassifier() {}

    public static TrainPersonDetectionClassifier create(AssetManager assetManager) throws Exception {
        if (classifier != null) return classifier;

        classifier = new TrainPersonDetectionClassifier();

        classifier.mtcnn = MTCNN.create(assetManager);
        classifier.faceNet = FaceNet.create(assetManager, FACE_SIZE, FACE_SIZE);
        classifier.svm = LibSVM.getInstance();

        classifier.classNames = FileUtils.readLabel(FileUtils.LABEL_FILE);

        return classifier;
    }

    public CharSequence[] getClassNames() {
        CharSequence[] cs = new CharSequence[classNames.size() + 1];
        int idx = 1;

        cs[0] = "+ Add new person";
        for (String name : classNames) {
            cs[idx++] = name;
        }

        return cs;
    }

    public void updateData(int label, ContentResolver contentResolver, ArrayList<Uri> uris) throws Exception {
        synchronized (this) {
            ArrayList<float[]> list = new ArrayList<>();

            for (Uri uri : uris) {
                Bitmap bitmap = getBitmapFromUri(contentResolver, uri);
                Pair faces[] = mtcnn.detect(bitmap);

                float max = 0f;
                Rect rect = new Rect();

                for (Pair face : faces) {
                    Float prob = (Float) face.second;
                    if (prob > max) {
                        max = prob;

                        RectF rectF = (RectF) face.first;
                        rectF.round(rect);
                    }
                }

                float[] emb_array = new float[EMBEDDING_SIZE];
                faceNet.getEmbeddings(bitmap, rect).get(emb_array);
                list.add(emb_array);
            }

            svm.train(label, list);
        }
    }

    public void updateData(String name, Bitmap bitmap) throws Exception {
        synchronized (this) {
            int label = addPerson(name);
            ArrayList<float[]> list = new ArrayList<>();
            Pair faces[] = mtcnn.detect(bitmap);

            float max = 0f;
            Rect rect = new Rect();

            for (Pair face : faces) {
                Float prob = (Float) face.second;
                if (prob > max) {
                    max = prob;

                    RectF rectF = (RectF) face.first;
                    rectF.round(rect);
                }
            }

            float[] emb_array = new float[EMBEDDING_SIZE];
            faceNet.getEmbeddings(bitmap, rect).get(emb_array);
            list.add(emb_array);
            svm.train(label, list);
        }
    }

    public void updateData(String name, ContentResolver contentResolver, Uri uri) throws Exception {
        updateData(name, getBitmapFromUri(contentResolver, uri));
    }

    public int addPerson(String name) {
        int index = getClassNameIndex(name);
        if(index>=0){
            return index;
        } else {
            FileUtils.appendText(name, FileUtils.LABEL_FILE);
            classNames.add(name);

            return classNames.size()-1;
        }
    }

    private int getClassNameIndex(String name) {
        for(int i=0;i<classNames.size();i++){
            if(name.equalsIgnoreCase(classNames.get(i)))
                return i;
        }
        return -1;
    }

    public void close() {
        mtcnn.close();
        faceNet.close();
    }

}
