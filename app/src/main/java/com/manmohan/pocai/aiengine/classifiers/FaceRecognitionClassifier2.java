/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
import android.util.Log;

import com.manmohan.pocai.aiengine.Classifier;
import com.manmohan.pocai.aiengine.env.FileUtils;
import com.manmohan.pocai.aiengine.wrapper.FaceNet;
import com.manmohan.pocai.aiengine.wrapper.LibSVM;
import com.manmohan.pocai.aiengine.wrapper.MTCNN;

import java.io.FileDescriptor;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import androidx.core.util.Pair;

/**
 * Generic interface for interacting with different recognition engines.
 */
public class FaceRecognitionClassifier2 implements Classifier{


    public static final int EMBEDDING_SIZE = 512;
    private static FaceRecognitionClassifier2 classifier;

    private MTCNN mtcnn;
    private FaceNet faceNet;
    private LibSVM svm;

    private List<String> classNames;

    private FaceRecognitionClassifier2() {}

    public static FaceRecognitionClassifier2 getInstance(AssetManager assetManager,
                                                        int inputHeight,
                                                        int inputWidth) throws Exception {
        if (classifier != null) return classifier;

        classifier = new FaceRecognitionClassifier2();

        classifier.mtcnn = MTCNN.create(assetManager);
        classifier.faceNet = FaceNet.create(assetManager, inputHeight, inputWidth);
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

    public List<Recognition> recognizeImage(Bitmap bitmap, Matrix matrix) {
        synchronized (this) {
            Pair faces[] = mtcnn.detect(bitmap);

            final List<Recognition> mappedRecognitions = new LinkedList<>();

            for (Pair face : faces) {
                RectF rectF = (RectF) face.first;

                Rect rect = new Rect();
                rectF.round(rect);

                FloatBuffer buffer = faceNet.getEmbeddings(bitmap, rect);

                Pair<Integer, Float> pair = svm.predict(buffer);
                Log.e("pair_", pair.toString());

                matrix.mapRect(rectF);
                Float prob = pair.second;

                String name;
                if (prob > 0.5)
                    name = classNames.get(pair.first);
                else
                    name = "Unknown";

                Recognition result =
                        new Recognition("" + pair.first, name, prob, rectF);
                mappedRecognitions.add(result);
            }
            return mappedRecognitions;
        }

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

    public int addPerson(String name) {
        FileUtils.appendText(name, FileUtils.LABEL_FILE);
        classNames.add(name);

        return classNames.size();
    }

    private Bitmap getBitmapFromUri(ContentResolver contentResolver, Uri uri) throws Exception {
        ParcelFileDescriptor parcelFileDescriptor =
                contentResolver.openFileDescriptor(uri, "r");
        FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
        Bitmap bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor);
        parcelFileDescriptor.close();

        return bitmap;
    }

    public void enableStatLogging(final boolean debug){
    }

    public String getStatString() {
        return faceNet.getStatString();
    }

    public void close() {
        mtcnn.close();
        faceNet.close();
    }
}
