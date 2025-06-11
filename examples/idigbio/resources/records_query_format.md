# iDigBio Query Format

The iDigBio query format is intended to be an easy to write query format for our APIs. Its basic structure consists
of a JSON dictionary, with zero or more top-level keys that reference fields in our indexes. (See: [Index Fields](
Index-Fields))

A basic multi-field query looks like this:

```json
{
  "scientificname": {
    "type": "exists"
  },
  "family": "asteraceae"
}
```

That query will look for anything in the family Asteraceae that also has the scientific name field populated.

Multiple fields are combined with the "AND" operator.

More details on the query types supported below. Note, the query types under all fields should work on any field in
the index, including the non-string fields. The query types under the other sections will only work with fields of
the matching type.

## All Fields

### Searching for a field being present in the record

```json
{
  "scientificname": {
    "type": "exists"
  }
}
```

### Searching for a field being absent in the record

```json
{
  "scientificname": {
    "type": "missing"
  }
}
```

### Full text searching

```json
{
  "data": {
    "type": "fulltext",
    "value": "aster"
  }
}
```

### Searching for a value within a field

```json
{
  "family": "asteraceae"
}
```

### Searching for multiple values within a field

```json
{
  "family": [
    "asteraceae",
    "fagaceae"
  ]
}
```

### Searching for a value by prefix

```json
{
  "family": {
    "type": "prefix",
    "value": "aster"
  }
}
```

## Boolean Fields

### Searching for a boolean value

```json
{
  "hasImage": true
}
```

## Numeric Fields

### Searching within a range for a numeric field

```json
{
  "minelevation": {
    "type": "range",
    "gte": "100",
    "lte": "200"
  }
}
```

## Date Fields

### Searching within a range on a date field

```json
{
  "datecollected": {
    "type": "range",
    "gte": "1800-01-01",
    "lte": "1900-01-01"
  }
}
```

## Geographic Point Fields

### Searching within a bounding box on a point field

```json
{
  "geopoint": {
    "type": "geo_bounding_box",
    "top_left": {
      "lat": 19.23,
      "lon": -130
    },
    "bottom_right": {
      "lat": -45.1119,
      "lon": 179.99999
    }
  }
}
```

### Searching within a radius around a geopoint

```json
{
  "geopoint": {
    "type": "geo_distance",
    "distance": "100km",
    "lat": -41.1119,
    "lon": 145.323
  }
}
```
