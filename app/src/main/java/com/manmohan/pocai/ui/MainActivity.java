package com.manmohan.pocai.ui;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

import com.manmohan.pocai.R;
import com.manmohan.pocai.ui.ext.CameraActivity;
import com.manmohan.pocai.ui.facerecognition.FaceRecognitionActivity;
import com.manmohan.pocai.ui.motiondetection.MotionDetectionActivity;
import com.manmohan.pocai.ui.trainfacerecognition.TrainFaceRecognition;
import com.manmohan.pocai.ui.trainfacerecognition.TrainFaceRecognitionActivity;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSIONS_REQUEST = 1;

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findViewById(R.id.face_recognition).setOnClickListener(v-> openFaceRecognition());
        findViewById(R.id.train_face_recognition).setOnClickListener(v-> openTrainFaceRecognition());
        findViewById(R.id.motion_detection).setOnClickListener(v-> openMotionDetection());

        if (!hasPermission()) {
            requestPermission();
        }
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
            } else {
                requestPermission();
            }
        }
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
                    shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
        }
    }

    private void openFaceRecognition() {
        startActivity(new Intent(this, FaceRecognitionActivity.class));
    }

    private void openTrainFaceRecognition() {
        startActivity(new Intent(this, TrainFaceRecognitionActivity.class));
//        startActivity(new Intent(this, TrainFaceRecognition.class));
    }

    private void openMotionDetection() {
        startActivity(new Intent(this, MotionDetectionActivity.class));

    }

}
