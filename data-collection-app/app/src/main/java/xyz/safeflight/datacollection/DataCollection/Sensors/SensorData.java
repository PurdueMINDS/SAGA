package xyz.safeflight.datacollection.DataCollection.Sensors;


import xyz.safeflight.datacollection.Utils;

/**
 * Created by migue on 9/21/2016.
 */
public class SensorData {
    private static SensorData instance = null;

    private Float acc_x = null;
    private Float acc_y = null;
    private Float acc_z = null;
    private String acc_last = "";

    private Float bar_pressure = null;
    private String bar_last = "";


    private Float gyr_x = null;
    private Float gyr_y = null;
    private Float gyr_z = null;
    private String gyr_last = "";

    public static void setInstance(SensorData instance) {
        SensorData.instance = instance;
    }

    private SensorData(){
    }

    // Method to maintain singleton instantiation
    public static SensorData getInstance() {
        if(instance == null) {
            instance = new SensorData();
        }
        return instance;
    }

    public void setAccValues(Float x, Float y, Float z, Long millis){
        acc_x = x;
        acc_y = y;
        acc_z = z;
        acc_last = Utils.formatTime(millis);
    }

    public void setGyroscopeValues(Float x, Float y, Float z, Long millis){
        gyr_x = x;
        gyr_y = y;
        gyr_z = z;
        gyr_last = Utils.formatTime(millis);
    }

    public void setBarometerValue(Float pressure,Long millis){
        bar_pressure = pressure;
        bar_last = Utils.formatTime(millis);
    }


    public Float getAcc_x() {
        return acc_x;
    }

    public void setAcc_x(Float acc_x) {
        this.acc_x = acc_x;
    }

    public Float getAcc_y() {
        return acc_y;
    }

    public void setAcc_y(Float acc_y) {
        this.acc_y = acc_y;
    }

    public Float getAcc_z() {
        return acc_z;
    }

    public void setAcc_z(Float acc_z) {
        this.acc_z = acc_z;
    }

    public String getAcc_last() {
        return acc_last;
    }

    public void setAcc_last(String acc_last) {
        this.acc_last = acc_last;
    }

    public Float getBar_pressure() {
        return bar_pressure;
    }

    public void setBar_pressure(Float bar_pressure) {
        this.bar_pressure = bar_pressure;
    }

    public String getBar_last() {
        return bar_last;
    }

    public void setBar_last(String bar_last) {
        this.bar_last = bar_last;
    }

    public Float getGyr_x() {
        return gyr_x;
    }

    public void setGyr_x(Float gyr_x) {
        this.gyr_x = gyr_x;
    }

    public Float getGyr_y() {
        return gyr_y;
    }

    public void setGyr_y(Float gyr_y) {
        this.gyr_y = gyr_y;
    }

    public Float getGyr_z() {
        return gyr_z;
    }

    public void setGyr_z(Float gyr_z) {
        this.gyr_z = gyr_z;
    }

    public String getGyr_last() {
        return gyr_last;
    }

    public void setGyr_last(String gyr_last) {
        this.gyr_last = gyr_last;
    }
}
