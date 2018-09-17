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

import com.opencsv.CSVWriter;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.Writer;
import java.lang.reflect.Field;
import java.util.LinkedList;
import java.util.List;

import timber.log.Timber;


class AHRSData {
    double GPSLastFixSinceMidnightUTC;
    double GPSLatitude;
    double GPSLongitude;
    double GPSFixQuality;
    double GPSHeightAboveEllipsoid;
    double GPSGeoidSep;
    double GPSSatellites;
    double GPSSatellitesTracked;
    double GPSSatellitesSeen;
    double GPSHorizontalAccuracy;
    double GPSNACp;
    double GPSAltitudeMSL;
    double GPSVerticalAccuracy;
    double GPSVerticalSpeed;
    String GPSLastFixLocalTime;
    double GPSTrueCourse;
    double GPSTurnRate;
    double GPSGroundSpeed;
    String GPSLastGroundTrackTime;
    String GPSTime;
    String GPSLastGPSTimeStratuxTime;
    String GPSLastValidNMEAMessageTime;
    String GPSLastValidNMEAMessage;
    double GPSPositionSampleRate;
    double BaroTemperature;
    double BaroPressureAltitude;
    double BaroVerticalSpeed;
    String BaroLastMeasurementTime;
    double AHRSPitch;
    double AHRSRoll;
    double AHRSGyroHeading;
    double AHRSMagHeading;
    double AHRSSlipSkid;
    double AHRSTurnRate;
    double AHRSGLoad;
    double AHRSGLoadMin;
    double AHRSGLoadMax;
    String AHRSLastAttitudeTime;
    double AHRSStatus;


    private static final String[] fields = {
        "GPSLastFixSinceMidnightUTC",
        "GPSLatitude",
        "GPSLongitude",
        "GPSFixQuality",
        "GPSHeightAboveEllipsoid",
        "GPSGeoidSep",
        "GPSSatellites",
        "GPSSatellitesTracked",
        "GPSSatellitesSeen",
        "GPSHorizontalAccuracy",
        "GPSNACp",
        "GPSAltitudeMSL",
        "GPSVerticalAccuracy",
        "GPSVerticalSpeed",
        "GPSLastFixLocalTime",
        "GPSTrueCourse",
        "GPSTurnRate",
        "GPSGroundSpeed",
        "GPSLastGroundTrackTime",
        "GPSTime",
        "GPSLastGPSTimeStratuxTime",
        "GPSLastValidNMEAMessageTime",
        "GPSLastValidNMEAMessage",
        "GPSPositionSampleRate",
        "BaroTemperature",
        "BaroPressureAltitude",
        "BaroVerticalSpeed",
        "BaroLastMeasurementTime",
        "AHRSPitch",
        "AHRSRoll",
        "AHRSGyroHeading",
        "AHRSMagHeading",
        "AHRSSlipSkid",
        "AHRSTurnRate",
        "AHRSGLoad",
        "AHRSGLoadMin",
        "AHRSGLoadMax",
        "AHRSLastAttitudeTime",
        "AHRSStatus"
    };

    AHRSData(JSONObject json) throws JSONException {
        GPSLastFixSinceMidnightUTC = json.getDouble("GPSLastFixSinceMidnightUTC");
        GPSLatitude = json.getDouble("GPSLatitude");
        GPSLongitude = json.getDouble("GPSLongitude");
        GPSFixQuality = json.getDouble("GPSFixQuality");
        GPSHeightAboveEllipsoid = json.getDouble("GPSHeightAboveEllipsoid");
        GPSGeoidSep = json.getDouble("GPSGeoidSep");
        GPSSatellites = json.getDouble("GPSSatellites");
        GPSSatellitesTracked = json.getDouble("GPSSatellitesTracked");
        GPSSatellitesSeen = json.getDouble("GPSSatellitesSeen");
        GPSHorizontalAccuracy = json.getDouble("GPSHorizontalAccuracy");
        GPSNACp = json.getDouble("GPSNACp");
        GPSAltitudeMSL = json.getDouble("GPSAltitudeMSL");
        GPSVerticalAccuracy = json.getDouble("GPSVerticalAccuracy");
        GPSVerticalSpeed = json.getDouble("GPSVerticalSpeed");
        GPSLastFixLocalTime = json.getString("GPSLastFixLocalTime");
        GPSTrueCourse = json.getDouble("GPSTrueCourse");
        GPSTurnRate = json.getDouble("GPSTurnRate");
        GPSGroundSpeed = json.getDouble("GPSGroundSpeed");
        GPSLastGroundTrackTime = json.getString("GPSLastGroundTrackTime");
        GPSTime = json.getString("GPSTime");
        GPSLastGPSTimeStratuxTime = json.getString("GPSLastGPSTimeStratuxTime");
        GPSLastValidNMEAMessageTime = json.getString("GPSLastValidNMEAMessageTime");
        GPSLastValidNMEAMessage = json.getString("GPSLastValidNMEAMessage");
        GPSPositionSampleRate = json.getDouble("GPSPositionSampleRate");
        BaroTemperature = json.getDouble("BaroTemperature");
        BaroPressureAltitude = json.getDouble("BaroPressureAltitude");
        BaroVerticalSpeed = json.getDouble("BaroVerticalSpeed");
        BaroLastMeasurementTime = json.getString("BaroLastMeasurementTime");
        AHRSPitch = json.getDouble("AHRSPitch");
        AHRSRoll = json.getDouble("AHRSRoll");
        AHRSGyroHeading = json.getDouble("AHRSGyroHeading");
        AHRSMagHeading = json.getDouble("AHRSMagHeading");
        AHRSSlipSkid = json.getDouble("AHRSSlipSkid");
        AHRSTurnRate = json.getDouble("AHRSTurnRate");
        AHRSGLoad = json.getDouble("AHRSGLoad");
        AHRSGLoadMin = json.getDouble("AHRSGLoadMin");
        AHRSGLoadMax = json.getDouble("AHRSGLoadMax");
        AHRSLastAttitudeTime = json.getString("AHRSLastAttitudeTime");
        AHRSStatus = json.getDouble("AHRSStatus");
    }

    JSONObject toJSON() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("GPSLastFixSinceMidnightUTC", GPSLastFixSinceMidnightUTC);
        json.put("GPSLatitude", GPSLatitude);
        json.put("GPSLongitude", GPSLongitude);
        json.put("GPSFixQuality", GPSFixQuality);
        json.put("GPSHeightAboveEllipsoid", GPSHeightAboveEllipsoid);
        json.put("GPSGeoidSep", GPSGeoidSep);
        json.put("GPSSatellites", GPSSatellites);
        json.put("GPSSatellitesTracked", GPSSatellitesTracked);
        json.put("GPSSatellitesSeen", GPSSatellitesSeen);
        json.put("GPSHorizontalAccuracy", GPSHorizontalAccuracy);
        json.put("GPSNACp", GPSNACp);
        json.put("GPSAltitudeMSL", GPSAltitudeMSL);
        json.put("GPSVerticalAccuracy", GPSVerticalAccuracy);
        json.put("GPSVerticalSpeed", GPSVerticalSpeed);
        json.put("GPSLastFixLocalTime", GPSLastFixLocalTime);
        json.put("GPSTrueCourse", GPSTrueCourse);
        json.put("GPSTurnRate", GPSTurnRate);
        json.put("GPSGroundSpeed", GPSGroundSpeed);
        json.put("GPSLastGroundTrackTime", GPSLastGroundTrackTime);
        json.put("GPSTime", GPSTime);
        json.put("GPSLastGPSTimeStratuxTime", GPSLastGPSTimeStratuxTime);
        json.put("GPSLastValidNMEAMessageTime", GPSLastValidNMEAMessageTime);
        json.put("GPSLastValidNMEAMessage", GPSLastValidNMEAMessage);
        json.put("GPSPositionSampleRate", GPSPositionSampleRate);
        json.put("BaroTemperature", BaroTemperature);
        json.put("BaroPressureAltitude", BaroPressureAltitude);
        json.put("BaroVerticalSpeed", BaroVerticalSpeed);
        json.put("BaroLastMeasurementTime", BaroLastMeasurementTime);
        json.put("AHRSPitch", AHRSPitch);
        json.put("AHRSRoll", AHRSRoll);
        json.put("AHRSGyroHeading", AHRSGyroHeading);
        json.put("AHRSMagHeading", AHRSMagHeading);
        json.put("AHRSSlipSkid", AHRSSlipSkid);
        json.put("AHRSTurnRate", AHRSTurnRate);
        json.put("AHRSGLoad", AHRSGLoad);
        json.put("AHRSGLoadMin", AHRSGLoadMin);
        json.put("AHRSGLoadMax", AHRSGLoadMax);
        json.put("AHRSLastAttitudeTime", AHRSLastAttitudeTime);
        json.put("AHRSStatus", AHRSStatus);

        return json;
    }

    private String[] toArray() {
        String[] array = new String[fields.length];

        try {
            for (int i = 0; i < fields.length; i++) {
                String fieldName = fields[i];
                Field field = AHRSData.class.getField(fieldName);
                array[i] = field.get(this).toString();
            }
        } catch (NoSuchFieldException e) {
            Timber.e(e, "Unnexpected field in toList()");
        } catch (IllegalAccessException e) {
            Timber.e(e, "Unnexpected field in toList()");
        }

        return array;
    }

    static void dumpCSV(Writer sink, List<AHRSData> objects) {
        dumpCSV(sink, objects, false);
    }

    static void dumpCSV(Writer sink, List<AHRSData> objects, boolean header) {
        CSVWriter writer = new CSVWriter(sink);

        List<String[]> toWrite = new LinkedList<>();

        if (header) {
            toWrite.add(fields);
        }

        if (objects != null) {
            for (AHRSData object : objects) {
                toWrite.add(object.toArray());
            }
        }

        writer.writeAll(toWrite);
    }
}
