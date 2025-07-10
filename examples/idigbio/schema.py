"""
This module provides pydantic schema used to craft and validate iDigBio API queries.
"""
from datetime import date
from typing import Optional, List, Union, Literal

from pydantic import Field, BaseModel, field_validator
from pydantic_core import PydanticCustomError


class DateRange(BaseModel):
    type: Literal["range"]
    gte: Optional[date] = Field(None, description="The start date of the range",
                                examples=["1900-3-14", "2024-01-01"])
    lte: Optional[date] = Field(None, description="The end date of the range.",
                                examples=["1900-12-20", "2024-02-01"])


class Existence(BaseModel):
    type: Literal["exists", "missing"]


Date = Union[date, DateRange, Existence]
String = Union[str, Existence]
Bool = Union[bool, Existence]
Float = Union[float, Existence]
Int = Union[int, Existence]


class Coordinate(BaseModel):
    """
    Represents a geographic coordinate with latitude and longitude.
    """
    lat: float = Field(..., description="latitude")
    lon: float = Field(..., description="longitude")

    @field_validator('lat', mode='after')
    @classmethod
    def validate_latitude(cls, v):
        if v is None:
            return v
        if not (-90 <= v <= 90):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid latitude value: {latitude} is not in range [-90, +90]",
                dict(latitude=v, terminal=True)
            )
        return v

    @field_validator('lon', mode='after')
    @classmethod
    def validate_longitude(cls, v):
        if v is None:
            return v
        if not (-180 <= v <= 180):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid longitude value: {longitude} is not in range [-180, +180]",
                dict(longitude=v, terminal=True)
            )
        return v


class GeoPoint(BaseModel):
    """
    This schema represents a location on earth.
    Supports two types:
    - geo_distance: A point with optional distance radius
    - geo_bounding_box: A rectangular area defined by top-left and bottom-right coordinates
    """
    type: Literal["geo_distance", "geo_bounding_box"] = Field(default="geo_distance")

    # Fields for geo_distance type
    lat: Optional[float] = Field(None, description="latitude (used only when type is geo_distance)")
    lon: Optional[float] = Field(None, description="longitude (used only when type is geo_distance)")
    distance: Optional[str] = Field(None,
                                    description="distance in kilometers with km at the end. Example: 575km (used only when type is geo_distance)")

    # Fields for geo_bounding_box type
    top_left: Optional[Coordinate] = Field(None,
                                           description="Top-left coordinate of bounding box (used only when type is geo_bounding_box)")
    bottom_right: Optional[Coordinate] = Field(None,
                                               description="Bottom-right coordinate of bounding box (used only when type is geo_bounding_box)")

    @field_validator('lat', mode='after')
    @classmethod
    def validate_latitude(cls, v, info):
        if v is None:
            return v
        if not (-90 <= v <= 90):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid latitude value: {latitude} is not in range [-90, +90]",
                dict(latitude=v, terminal=True)
            )
        return v

    @field_validator('lon', mode='after')
    @classmethod
    def validate_longitude(cls, v, info):
        if v is None:
            return v
        if not (-180 <= v <= 180):
            raise PydanticCustomError(
                "geopoint_range_error",
                "Error: Invalid longitude value: {longitude} is not in range [-180, +180]",
                dict(longitude=v, terminal=True)
            )
        return v

    @field_validator('top_left', 'bottom_right', mode='after')
    @classmethod
    def validate_coordinates_for_type(cls, v, info):
        # Make sure coordinate fields are only present when type is geo_bounding_box
        if info.data.get('type') == 'geo_bounding_box' and v is None and info.field_name in info.data:
            raise PydanticCustomError(
                "geo_missing_field",
                "Error: {field_name} is required when type is geo_bounding_box",
                dict(field_name=info.field_name, terminal=True)
            )
        return v

    @field_validator('lat', 'lon', 'distance', mode='after')
    @classmethod
    def validate_fields_for_type(cls, v, info):
        # Make sure distance point fields are only present when type is geo_distance
        if info.data.get('type') == 'geo_distance' and info.field_name in ['lat', 'lon'] and v is None:
            raise PydanticCustomError(
                "geo_missing_field",
                "Error: {field_name} is required when type is geo_distance",
                dict(field_name=info.field_name, terminal=True)
            )
        return v

    # Model-level validator to check overall consistency
    @field_validator('type')
    @classmethod
    def validate_model_consistency(cls, v, info):
        if v == 'geo_distance':
            # When geo_distance, top_left and bottom_right should not be present
            if info.data.get('top_left') is not None or info.data.get('bottom_right') is not None:
                raise PydanticCustomError(
                    "geo_type_mismatch",
                    "Error: top_left and bottom_right should not be present when type is geo_distance",
                    dict(terminal=True)
                )
        elif v == 'geo_bounding_box':
            # When geo_bounding_box, lat, lon, and distance should not be present
            if any(info.data.get(field) is not None for field in ['lat', 'lon', 'distance']):
                raise PydanticCustomError(
                    "geo_type_mismatch",
                    "Error: lat, lon, and distance should not be present when type is geo_bounding_box",
                    dict(terminal=True)
                )
        return v


class IDBRecordsQuerySchema(BaseModel):
    """
    This schema represents the iDigBio Record Query Format.
    """
    associatedsequences: Optional[String] = None
    barcodevalue: Optional[String] = None
    basisofrecord: Optional[String] = None
    bed: Optional[String] = Field(None,
                                  description="The full name of the lithostratigraphic bed from which a material entity was collected.")
    canonicalname: Optional[String] = Field(None,
                                            description="The latinized elements of a scientific name, without authorship information, etc.")
    catalognumber: Optional[String] = None
    class_: Optional[String] = Field(None, alias="class", description="The taxonomic class of an organism")
    collectioncode: Optional[String] = None
    collectionid: Optional[String] = None
    collectionname: Optional[String] = None
    collector: Optional[String] = None
    commonname: Optional[String] = Field(None,
                                         description="Common name for a specific species. Do not use for taxonomic "
                                                     "groups like \"birds\" or \"mammals\"")
    # commonnames
    continent: Optional[String] = None
    # coordinateuncertainty
    country: Optional[String] = Field(None,
                                      description="Full, accepted country name. For example 'Canada' instead of CAD.")
    # countrycode
    county: Optional[String] = None
    # data {}
    # datasetid
    datecollected: Optional[Date] = None
    datemodified: Optional[Date] = None
    dqs: Optional[Float] = Field(None, description="Data quality score for the record")
    # earliestageorloweststage
    # earliesteonorlowesteonothem
    # earliestepochorlowestseries
    # earliesteraorlowesterathem
    # earliestperiodorlowestsystem: Optional[String] = None
    etag: Optional[String] = None
    eventdate: Optional[Date] = None
    family: Optional[String] = None
    fieldnumber: Optional[String] = None
    flags: Optional[String] = None
    # formation
    genus: Optional[String] = None
    # geologicalcontextid
    geopoint: Optional[GeoPoint] = None
    # group
    hasImage: Optional[bool] = None  # All records have this field, no need to allow existence queries
    highertaxon: Optional[String] = None
    # highestbiostratigraphiczone
    # indexData {}
    # individualcount
    infraspecificepithet: Optional[String] = None
    institutioncode: Optional[String] = Field(None,
                                              description="The name (or acronym) in use by the institution having custody of the object(s) or information referred to in the record.")
    institutionid: Optional[String] = Field(None,
                                            description="An identifier for the institution having custody of the object(s) or information referred to in the record.")
    institutionname: Optional[String] = None
    kingdom: Optional[String] = None
    # latestageorhigheststage
    # latesteonorhighesteonothem
    # latestepochorhighestseries
    # latesteraorhighesterathem
    # latestperiodorhighestsystem: Optional[String] = None
    # lithostratigraphicterms
    locality: Optional[String] = None
    # lowestbiostratigraphiczone
    maxdepth: Optional[Float] = None
    maxelevation: Optional[Float] = None
    mediarecords: Optional[String] = None
    # member
    mindepth: Optional[Float] = None
    minelevation: Optional[Float] = None
    municipality: Optional[String] = None
    occurrenceid: Optional[String] = None
    order: Optional[String] = None
    phylum: Optional[String] = None
    # query
    recordids: Optional[String] = None
    recordnumber: Optional[String] = None
    recordset: Optional[String] = None
    scientificname: Optional[Union[String, List[str]]] = None
    # size
    specificepithet: Optional[String] = None
    # startdayofyear
    stateprovince: Optional[Union[String, List[str]]] = None
    taxonid: Optional[String] = Field(None,
                                      description="An identifier for the set of dwc:Taxon information. May be a global unique identifier or an identifier specific to the data set.")
    taxonomicstatus: Optional[String] = Field(None,
                                              description="The status of the use of the scientificname as a label for a taxon",
                                              examples=["invalid", "misapplied", "homotypic synonym", "accepted"])
    taxonrank: Optional[String] = None
    typestatus: Optional[String] = Field(None,
                                         description="A list (concatenated and separated) of nomenclatural types (type status, typified scientific name, publication) applied to the subject.",
                                         examples=["holotype of Pinus abies | holotype of Picea abies"])
    uuid: Optional[String] = Field(None, description="An internal identifier used by iDigBio to identify the record")
    verbatimeventdate: Optional[String] = None
    verbatimlocality: Optional[String] = None
    version: Optional[Int] = None
    waterbody: Optional[String] = None

    class Config:
        json_encoders = {
            date: date.isoformat
        }


class IDBMediaQuerySchema(BaseModel):
    """
    This schema represents the iDigBio Media Query Format.
    """
    accessuri: Optional[String] = None
    datemodified: Optional[Date] = Field(None, description="The \"datemodified\" field in the original media record")
    # dqs: Optional[Float] = Field(None, description="Data quality score for the mediarecord. DO not use unless specified by user.")
    etag: Optional[String] = None
    # Should be a Literal, leave commented for now to prevent undefined behavior. 
    # flags: Optional[String] = None
    # format: Optional[String] = Field(None, description="Image format. Do not use this field unless the user specifies a format.")
    # All records have "hasSpecimen", no need to allow existence queries
    # hasSpecimen: Optional[bool] = Field(None,
    # description="Whether the media record is associated with a specific species "
    #             "occurrence record")
    licenselogourl: Optional[String] = None
    mediatype: Optional[Literal["images", "sounds"]] = None
    # modified: Optional[String] = None # TODO: how this is different from datemodified?
    modified: Optional[Date] = Field(None,
                                     description="Last time the media record changed in iDigBio, whether the original "
                                                 "record or iDigBio's metadata")
    recordids: Optional[String] = None
    records: Optional[String] = Field(None, description="UUIDs for records that are associated with the media record")
    recordset: Optional[String] = Field(None, description="The record set that the media record is a part of")
    rights: Optional[String] = None
    # tag: Optional[String] = None # TODO
    # type: Optional[String] = None
    uuid: Optional[String] = Field(None, description="An identifier used by iDigBio to identify the mediarecord")
    version: Optional[Int] = None

    # webstatement: Optional[String] = None # TODO
    # xpixels: Optional[Int] = None
    # ypixels: Optional[Int] = None

    class Config:
        json_encoders = {
            date: date.isoformat
        }


class IDigBioRecordsApiParameters(BaseModel):
    """
    This schema represents the output containing the LLM-generated iDigBio query. 
    """
    rq: IDBRecordsQuerySchema = Field(...,
                                      description="Search criteria for species occurrence records in iDigBio")
    limit: Optional[int] = Field(100, ge=1, le=5000,
                                 description="The maximum number of records to return")
