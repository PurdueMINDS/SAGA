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

package xyz.safeflight.datacollection.DataCollection;

import android.Manifest;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.IBinder;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.LocalBroadcastManager;
import android.util.Log;

import xyz.safeflight.datacollection.Constants;
import xyz.safeflight.datacollection.DataCollection.Sensors.SensorClass;
import xyz.safeflight.datacollection.DataCollection.Sensors.SensorData;
import xyz.safeflight.datacollection.Utils;

import java.util.ArrayList;

/**
 * Created by migue on 9/12/2016.
 */
public class LocationService extends Service {
    private static final String TAG = "LocationService";
    private LocationManager mLocationManager = null;
    private static long GPS_INTERVAL = 5*1000;
    private static final float LOCATION_DISTANCE = 0;
    private LocalBroadcastManager broadcaster;
    private BroadcastReceiver receiver;
    private Boolean commFlag = true;
    private Boolean plot = false;
    private ArrayList<DataObject> savedData;
    private SensorClass sensorManager;
    private int totalLocs = 0;

    /* Code for interacting with GPS inside the service and send messages to main thread with location
    updates in order to let him modify the UI
     */
    private class LocationListener implements android.location.LocationListener {
        Location mLastLocation;

        public LocationListener(String provider) {
            Log.e(TAG, "LocationListener " + provider);
            mLastLocation = new Location(provider);
        }

        @Override
        public void onLocationChanged(Location location) {
            //Log.e(TAG, "onLocationChanged: " + location);
            mLastLocation.set(location);
            totalLocs += 1;

            // Send plot data to PlotActivity
            DataObject tmp = processNewLocation(location, totalLocs);

            if (!commFlag) {
                Log.e(TAG,"NO COMM!! SAVING...");
                // Save data into arraylist while activity is not on foreground
                savedData.add(tmp);
                if (plot){
                    sendPlotData(tmp);
                }
            } else {
                // Send data to CollectData
                sendNewData(tmp);
            }

            Log.e(TAG,"ID OF LOCATION: "+tmp.getID());
        }

        @Override
        public void onProviderDisabled(String provider) {
            Log.e(TAG, "onProviderDisabled: " + provider);
            Intent intent = new Intent(Constants.TURN_ON_LOC);
            broadcaster.sendBroadcast(intent);
        }

        @Override
        public void onProviderEnabled(String provider) {
            Log.e(TAG, "onProviderEnabled: " + provider);
        }

        @Override
        public void onStatusChanged(String provider, int status, Bundle extras) {
            Log.e(TAG, "onStatusChanged: " + provider);
        }

    }

    LocationListener locationListener = new LocationListener(LocationManager.GPS_PROVIDER);

    @Override
    public IBinder onBind(Intent arg0) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        super.onStartCommand(intent, flags, startId);

        // Get GPS update frequency from Intent of previous activity
        GPS_INTERVAL = (long) intent.getIntExtra("period",0) * 1000;

        // Start gps listener class
        try {
            mLocationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER, GPS_INTERVAL, LOCATION_DISTANCE, locationListener);
        } catch (java.lang.SecurityException ex) {
            Log.i(TAG, "fail to request location update, ignore", ex);

        } catch (IllegalArgumentException ex) {
            Log.d(TAG, "gps provider does not exist " + ex.getMessage());
        }

        // Create sensor listener class
        sensorManager = new SensorClass(this,GPS_INTERVAL);

        // Start bd manager to communicate with main thread
        IntentFilter myfilter = new IntentFilter();
        myfilter.addAction(Constants.NO_COMMUNICATION);
        myfilter.addAction(Constants.BACK_TO_WORK);
        myfilter.addAction(Constants.PLOT);
        // Register the receiver
        LocalBroadcastManager.getInstance(this).registerReceiver((receiver),myfilter);

        return START_STICKY;
    }

    @Override
    public void onCreate() {
        Log.e(TAG, "onCreate Service");
        // Initialize broadcast communication with main
        broadcaster = LocalBroadcastManager.getInstance(this);

        // Register receiver to check when Activity in onStop
        receiver = new BroadcastReceiver(){
            @Override
            public void onReceive(Context context, Intent intent) {
                processMessage(intent);
            }
        };

        initializeLocationManager();


    }

    @Override
    public void onDestroy() {
        Log.e(TAG, "onDestroy Service");
        super.onDestroy();
        if (mLocationManager != null) {
            try {
                if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
                    Log.i(TAG,"problem on destroy no permission for this");
                }
                mLocationManager.removeUpdates(locationListener);
            } catch (Exception ex) {
                Log.i(TAG, "fail to remove location listeners, ignore", ex);
            }
        }
        sensorManager.unRegisterListeners();
    }

    private void initializeLocationManager() {
        if (mLocationManager == null) {
            mLocationManager = (LocationManager) getApplicationContext().getSystemService(Context.LOCATION_SERVICE);
        }
    }

    private void processMessage(Intent intent){
        String action = intent.getAction();
        switch (action){
            case Constants.NO_COMMUNICATION:
                Log.e(TAG,"Received activity is onstop message");
                commFlag = false;
                savedData = new ArrayList<>();
                break;
            case Constants.BACK_TO_WORK:
                Log.e(TAG,"Received activity has come back message");
                commFlag = true;
                plot = false;
                sendResume();
                break;
            case Constants.PLOT:
                Log.e(TAG, "Received plotting activity is starting");
                plot = true;
                commFlag = false;
                savedData = new ArrayList<>();
                break;
        }
    }


    private DataObject processNewLocation(Location location, int msg_id){
        // Get time
        String timestamp = Utils.getTime("HH:mm:ss");
        // Add last location to data object
        DataObject newdata = new DataObject(SensorData.getInstance(), location, timestamp, msg_id);
        //sendNewData(tmp); this is now in onLocationChanged
        return newdata;
    }

    private void sendNewData(DataObject data){
        Intent intent = new Intent(Constants.RESULT);
        if (data != null) {
            intent.putExtra(Constants.MESSAGE, data);
        }
        broadcaster.sendBroadcast(intent);
    }

    private void sendPlotData(DataObject data) {
        Log.e(TAG, "Sending plot value to PLOT ACTIVITY");
        Intent intent = new Intent(Constants.PLOT_VALUE);
        if (data != null) {
            intent.putExtra(Constants.MESSAGE, data);
        }
        broadcaster.sendBroadcast(intent);
    }

    private void sendResume(){
        Intent intent = new Intent(Constants.RESUME);
        intent.putExtra(Constants.MESSAGE, savedData);
        broadcaster.sendBroadcast(intent);
    }

}