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

package xyz.safeflight.datacollection.DataCollection.Sensors;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import xyz.safeflight.datacollection.Constants;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by migue on 9/20/2016.
 */
public class SensorClass implements SensorEventListener {
    private final String TAG = "SensorClass";
    private SensorManager mSensorManager;
    private Context mycontext;
    private float updateFrequency;
    private Sensor accelerometer;
    private Sensor barometer;
    private Sensor gyroscope;
    // Array with last updates of each sensor: 0<-Acc, 1<-Bar, 2 <- Gyr
    private ArrayList<Long> lastUpdate;
    private SensorData data;


    public SensorClass(Context context, long gpsInterval) {
        mycontext = context;

        if (gpsInterval!=0) {
            updateFrequency = (gpsInterval / Constants.SENSORACCURACY); // Frequency is 4 times higher than gps
        } else {
            updateFrequency = 0;
        }
        lastUpdate = new ArrayList<Long>();
        for (int i = 0; i < 3; i++) {
            lastUpdate.add((long) 0);
        }

        initSensors();

    }


    @Override
    public void onSensorChanged(SensorEvent event) {
        Sensor sensor = event.sensor;
        long curTime = System.currentTimeMillis();

        switch (sensor.getType()) {

            case Sensor.TYPE_ACCELEROMETER:
                float x = event.values[0];
                float y = event.values[1];
                float z = event.values[2];
                // Check if time interval with last measure is large enough
                if (checkRate(0,curTime)) {
                    data.setAccValues(x, y, z, curTime);
                }
                break;

            case Sensor.TYPE_PRESSURE:
                float pressure = event.values[0];
                data.setBar_pressure(pressure);
                Log.e(TAG,"NEW PRESSURE MEASURE----->"+String.valueOf(pressure));
                break;

            case Sensor.TYPE_GYROSCOPE:
                float x_g = event.values[0];
                float y_g = event.values[1];
                float z_g = event.values[2];
                if (checkRate(2,curTime)) {
                    data.setGyroscopeValues(x_g,y_g,z_g,curTime);
                }
                break;

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    private Sensor getSensor(Integer sensorType){
        return mSensorManager.getDefaultSensor(sensorType);
    }

    public void unRegisterListeners(){
        mSensorManager.unregisterListener(this);
    }

    public ArrayList<Integer> initSensors(){
        mSensorManager = (SensorManager) mycontext.getSystemService(Context.SENSOR_SERVICE);
        accelerometer = getSensor(Sensor.TYPE_ACCELEROMETER);
        barometer = getSensor(Sensor.TYPE_PRESSURE);
        gyroscope = getSensor(Sensor.TYPE_GYROSCOPE);
        // State 0 if sensor exists and 1 otherwise
        Integer[] state = new Integer[4];

        data = SensorData.getInstance();

        if (accelerometer!= null){
            mSensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        } else {
            data.setAccValues(null,null,null,null);
            state[Constants.ACCELEROMETER] = 1;
        }

        if (barometer!=null){
            mSensorManager.registerListener(this, barometer, SensorManager.SENSOR_DELAY_NORMAL);
        } else {
            state[Constants.BAROMETER] = 1;
        }

        if (gyroscope!=null){
            mSensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_NORMAL);
        } else {
            state[Constants.GYROSCOPE] = 1;
        }
        return new ArrayList<Integer>(Arrays.<Integer>asList(state));
    }


    private boolean checkRate(int sensor, long curTime){
        long interval = curTime - lastUpdate.get(sensor);

        if (interval >= updateFrequency) {
            //Log.e(TAG, String.valueOf(interval));
            lastUpdate.set(sensor, curTime);
            return true;
        }
        return false;
    }

    public static ArrayList<Integer> sensorInfo(Context c){
        SensorManager mSensorManager = (SensorManager) c.getSystemService(Context.SENSOR_SERVICE);
         Sensor accelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Sensor barometer = mSensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE);
        Sensor gyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        // State 0 if sensor exists and 1 otherwise
        Integer[] state = new Integer[4];
        for (int i=0;i<4;i++){
            state[i]=0;
        }
        if (accelerometer== null){
            state[Constants.ACCELEROMETER] = 1;
        }

        if (barometer==null){
            state[Constants.BAROMETER] = 1;
        }

        if (gyroscope==null){
            state[Constants.GYROSCOPE] = 1;
        }
        return new ArrayList<Integer>(Arrays.<Integer>asList(state));
    }
}
