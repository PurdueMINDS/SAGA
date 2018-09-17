/*
 *  Copyright 2018 Jianfei Gao, Leonardo Teixeira, Bruno Ribeiro.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package xyz.safeflight.stratuxlogger;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.IBinder;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.w3c.dom.Text;

import butterknife.BindView;
import butterknife.ButterKnife;
import timber.log.Timber;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 1;

    @BindView(R.id.record_btn) Button record_btn;
    @BindView(R.id.status) TextView status_txt;
    @BindView(R.id.numCollected_txt) TextView numCollected_txt;
    @BindView(R.id.lastReceived_txt) TextView lastReceived_txt;
    private boolean mBound = false;

    private StratuxCollectorService mService;
    private ServiceConnection mConnection = new ServiceConnection() {
        // Called when the connection with the service is established
        public void onServiceConnected(ComponentName className, IBinder service) {
            // Because we have bound to an explicit
            // service that is running in our own process, we can
            // cast its IBinder to a concrete class and directly access it.
            Timber.d("onServiceConnected()");
            StratuxCollectorService.StratuxCollectorBinder binder = (StratuxCollectorService.StratuxCollectorBinder) service;
            mService = binder.getService();
            mBound = true;
            mService.registerCallback(new Runnable() {
                @Override
                public void run() {
                    updateUi();
                }
            });
            updateUi();
        }

        // Called when the connection with the service disconnects unexpectedly
        public void onServiceDisconnected(ComponentName className) {
            Timber.e("onServiceDisconnected()");
            mBound = false;
            updateUi();
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);

        Intent intent = new Intent(this, StratuxCollectorService.class);
        bindService(intent, mConnection, Context.BIND_AUTO_CREATE);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mBound && mConnection != null) {
            unbindService(mConnection);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    if (mBound) mService.startCollecting();
                } else {
                    Timber.w("Permission denied");
                }
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        updateUi();
    }

    private void showPermissionExplanation() {
        // 1. Instantiate an AlertDialog.Builder with its constructor
        AlertDialog.Builder builder = new AlertDialog.Builder(this);

        // 2. Chain together various setter methods to set the dialog characteristics
        builder.setTitle("Permission Needed")
                .setMessage("Permission to write to external storage is needed in order to save the " +
                        "data collected from Stratux.")
                .setNeutralButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        ActivityCompat.requestPermissions(MainActivity.this,
                                new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                                PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);
                    }
                });

        // 3. Get the AlertDialog from create()
        AlertDialog dialog = builder.create();

        dialog.show();
    }

    void requestWritePermission() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {

            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(
                    this, Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
                showPermissionExplanation();
            } else {
                // No explanation needed, we can request the permission.
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);
            }
        }
    }

    public void RecordButton_onClick(View v) {
        // There's no point in doing anything util the service is up and running.
        // Indeed, the button shouldn't even be enabled if that is not the case.
        if (mBound) {
            if (mService.isCollecting()) {
                mService.stopCollecting();
            }
            else {
                int writeExternalStoragePermission = ContextCompat.checkSelfPermission(
                        this,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                );

                if (writeExternalStoragePermission == PackageManager.PERMISSION_GRANTED) {
                    // Good, we have the permission. Just issue the start request
                    mService.startCollecting();
                }
                else {
                    // Ops... We don't have the permission... Must ask it first!
                    requestWritePermission();
                }
            }
        }
        else {
            Timber.w("Button was clicked, but service is not bound.");
        }
    }

    public void updateUi() {
        Timber.d("Updating UI");

        if (mBound) {
            if (mService.isCollecting()) {
                record_btn.setText(R.string.stop_recording);
                status_txt.setText(R.string.status_recording);
                status_txt.setTextColor(getResources().getColor(R.color.colorOKTxt));
            }
            else {
                record_btn.setText(R.string.start_recording);
                status_txt.setText(R.string.status_not_recording);
                status_txt.setTextColor(getResources().getColor(R.color.colorErrorTxt));
            }
            record_btn.setEnabled(true);
            numCollected_txt.setText(String.valueOf(mService.getNumCollected()));
            lastReceived_txt.setText(mService.getLastReceived());
        }
        else {
            record_btn.setEnabled(false);
            Timber.w("Collector Service is not bound.");
        }
    }

}
