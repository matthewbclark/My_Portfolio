#Matt Clark
#4/19/2018
#Purpose: Tool for data clean up process - Drops unwanted fields, Deletes fields owned by the City of Boulder, Deletes duplicate fields 

#Dropping unwanted fields
import arcpy, os
env = r"C:\Users\Matt\Documents\College_Assignments\Spring_2018\Python\Final_Project\TestSubjects"
pathway  = "C:\Users\Matt\Documents\College_Assignments\Spring_2018\Python\Final_Project\TestSubjects\Test_Parcel13.shp"

try: 
    arcpy.env.workspace = env
    fields = arcpy.ListFields(pathway)
    for field in fields:
        print field.name

    ApprovedFields = ["OWNER_NAME", "OWNER_ICO", "OWNER_ADDR", "OWNER_CITY", "OWNER_STA", "OWNER_ZIP", "OWNER_ZIP2", "Shape", "FID", "OID"]
    print ApprovedFields

    fieldNameList = []
    print "fieldNameList created"
    
    for field in fields:
        if not field.name in ApprovedFields:
            fieldNameList.append(field.name)
    print "PRINTING FIELDS IN FIELD NAME LIST"
    for field in fieldNameList:        
        print field
    
    arcpy.DeleteField_management(pathway, fieldNameList)
    print "PRINTING FINAL FIELDS"
    new_fields = arcpy.ListFields(pathway)
    for field in new_fields:
        print field.name
    
except Exception as err:
    print(err.args[0])

#Deleting parcels owned by the City of Boulder and Duplicates.
try: 
    arcpy.env.workspace = env
    fields = arcpy.ListFields(pathway)
    for field in fields:
        print field.name
except:
    print e.message

try: 
    file = pathway
    clause = arcpy.AddFieldDelimiters(file, "OWNER_NAME") + " LIKE 'CITY OF BOULDER%' AND "+arcpy.AddFieldDelimiters(file, "OWNER_NAME") + " NOT LIKE '%HOUSING AUTHORITY'"
    
    
    with arcpy.da.UpdateCursor(file, ("OWNER_NAME"), clause) as cursor:
        cntr = 1
        for row in cursor:
            cursor.deleteRow()
            print "Record number " + str(cntr) + "deleted"
            cntr = cntr + 1
            
    #Deleting Duplicates             
    savedRows = []

    with arcpy.da.UpdateCursor(file, ("OWNER_NAME")) as cursor:
        cntr = 1
        for row in cursor:
            if row in savedRows:
                cursor.deleteRow()
                print "Record number " + str(cntr) + " deleted."
                cntr = cntr + 1
            else:
                savedRows.append(row)

except Exception as e:
    print e.message



