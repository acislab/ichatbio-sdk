If the user places a search term in quotes, always use the full text that is quoted, do not break it up.

If the user specifies an unquoted binomial species name (without additional authorship, variation, subspecies, etc.
information), try to break it up into its genus and specific epithet.

If the user specifies that they want to search for an exact scientific name, use the "scientificname" field.

## Example 1

```
Request: "Homo sapiens"
rq: {
    "genus": "Homo",
    "specificepithet": "sapiens"
}
```

## Example 2

```
Request: "Only Homo sapiens Linnaeus, 1758"
rq: {
    "scientificname": "Homo sapiens Linnaeus, 1758"
}
```

## Example 3 -

```
Request: "Scientific name \\"this is fake but use it anyway\\""
rq: {
    "scientificname": "this is fake but use it anyway"
}
```

## Example 4 - only records that specify a given field

```
Request: "kingdom must be specified"
rq: {
    "kingdom": {
        "type": "exists"
    }
}
```

## Example 5 - strings can be specified as lists

```
Request: "Homo sapiens and Rattus rattus in North America and Australia"
rq: {
    "scientificname": ["Homo sapiens", "Rattus rattus"],
    "continent": ["North America", "Australia"]
}
```

## Example 6 - only match records that specify a field

```
Request: "Records with a common name"
rq: {
    "commonname": {
        "type": "exists"
    }
}
```

## Example 7 - only match records that do NOT specify a field

```
Request: "Records with no family classification"
rq: {
    "family": {
        "type": "missing"
    }
}
```

## Example 8 - records with boolean fields

```
Request: "Records with no family classification"
rq: {
    "family": {
        "type": "missing"
    }
}
```