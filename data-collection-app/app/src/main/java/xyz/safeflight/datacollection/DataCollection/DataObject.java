package xyz.safeflight.datacollection.DataCollection;

import android.location.Location;
import android.os.Parcel;
import android.os.Parcelable;

import xyz.safeflight.datacollection.DataCollection.Sensors.SensorData;
import xyz.safeflight.datacollection.Utils;

/**
 * Created by migue on 9/21/2016.
 */
public class DataObject implements Parcelable {
    private int ID = 0;
    private Float acc_x = null;
    private Float acc_y = null;
    private Float acc_z = null;
    private String acc_last;
    private Float bar_pressure = null;
    private String bar_last = "";
    private Float gyr_x = null;
    private Float gyr_y = null;
    private Float gyr_z = null;
    private String gyr_last = "";
    private String timestamp;

    private Location location;

    public DataObject(SensorData d, Location location, String timestamp, Integer ID){
        this.location = location;
        this.timestamp = timestamp;
        this.ID = ID;
        copyData(d);

    }



    protected DataObject(Parcel in) {
        ID = in.readInt();
        acc_x = in.readFloat();
        acc_y = in.readFloat();
        acc_z = in.readFloat();
        acc_last = in.readString();
        bar_pressure = in.readFloat();
        bar_last = in.readString();
        gyr_x = in.readFloat();
        gyr_y = in.readFloat();
        gyr_z = in.readFloat();
        gyr_last = in.readString();
        location = (Location) in.readValue(Location.class.getClassLoader());
    }


    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeInt(ID);
        dest.writeFloat(acc_x);
        dest.writeFloat(acc_y);
        dest.writeFloat(acc_z);
        dest.writeString(acc_last);
        dest.writeFloat(bar_pressure);
        dest.writeString(bar_last);
        dest.writeFloat(gyr_x);
        dest.writeFloat(gyr_y);
        dest.writeFloat(gyr_z);
        dest.writeString(gyr_last);
        dest.writeValue(location);
    }


    @SuppressWarnings("unused")
    public static final Parcelable.Creator<DataObject> CREATOR = new Parcelable.Creator<DataObject>() {
        @Override
        public DataObject createFromParcel(Parcel in) {
            return new DataObject(in);
        }

        @Override
        public DataObject[] newArray(int size) {
            return new DataObject[size];
        }
    };


    @Override
    public String toString() {
        String ID_  = String.valueOf(ID) + "\t\t";
        String GPS = String.format("\t\t%f\t\t%f\t\t%f\t\t%f\t\t",location.getAltitude(),location.getLatitude(),location.getLongitude(),location.getAccuracy());
        String ACC = String.format("%f\t\t%f\t\t%f\t\t",acc_x,acc_y,acc_z);
        String GYR = String.format("%f\t\t%f\t\t%f\t\t",gyr_x,gyr_y,gyr_z);
        String BAR = String.format("%f\n",bar_pressure);

        /**
         * GPS Time, [by Jianfei Gao].
         */
        /* (Original)
        String loc = ID_ + timestamp + GPS + ACC + GYR + BAR;
        */
        // (Modified)
        String GTIME = new java.text.SimpleDateFormat("HH:mm:ss").format(location.getTime());
        String time_provider = location.getProvider();
        if (location.getProvider().equals(android.location.LocationManager.GPS_PROVIDER))
            android.util.Log.d("Location", "Time GPS: " + GTIME); // This is what we want!
        else
            android.util.Log.d("Location", "Time Device (" + location.getProvider() + "): " + GTIME);
        String timepackage = timestamp + "\t\t" + GTIME + "\t\t" + time_provider;
        String loc = ID_ + timepackage + GPS + ACC + GYR + BAR;
        // ---(Ending)---

        return loc;
    }

    private void copyData(SensorData d){
        acc_x = d.getAcc_x();
        acc_y = d.getAcc_y();
        acc_z = d.getAcc_z();
        acc_last = d.getAcc_last();
        gyr_x = d.getGyr_x();
        gyr_y = d.getGyr_y();
        gyr_z = d.getGyr_z();
        gyr_last = d.getGyr_last();
        bar_last = d.getBar_last();
        bar_pressure = d.getBar_pressure();
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

    public Location getLocation() {
        return location;
    }

    public void setLocation(Location location) {
        this.location = location;
    }

    public int getID() {
        return ID;
    }

    public void setID(int ID) {
        this.ID = ID;
    }
}
