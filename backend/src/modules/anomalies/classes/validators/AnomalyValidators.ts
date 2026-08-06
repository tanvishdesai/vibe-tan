import {
  IsArray,
  IsEmail,
  IsEnum,
  IsInt,
  IsMongoId,
  IsNotEmpty,
  IsNumber,
  IsOptional,
  IsString,
  Max,
  Min,
} from 'class-validator';
import {Transform, Type} from 'class-transformer';
import {JSONSchema} from 'class-validator-jsonschema';
// Import directly from the defining transformer rather than the module barrel
// to avoid a circular dependency (the barrel re-exports controllers that pull
// these validators back in, leaving AnomalyType undefined at eval time).
import {AnomalyType, FileType, IAnomalyData} from '../transformers/Anomaly.js';
import {ObjectId} from 'mongodb';
import {
  SortOrder,
  PaginationWithSortQuery,
} from '#root/shared/interfaces/models.js';

export class NewAnomalyData {
  @JSONSchema({
    description: 'The type of anomaly detected',
    example: AnomalyType.VOICE_DETECTION,
  })
  @IsEnum(AnomalyType)
  @IsNotEmpty()
  type: AnomalyType;

  @JSONSchema({
    description: 'Course ID associated with the anomaly',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  courseId: string | ObjectId;

  @JSONSchema({
    description: 'Version ID associated with the anomaly',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  versionId: string | ObjectId;

  @JSONSchema({
    description: 'Item ID associated with the anomaly',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  itemId: string | ObjectId;

  @JSONSchema({
    description: 'Cohort ID associated with the anomaly (optional)',
    type: 'string',
  })
  @IsOptional()
  @IsMongoId()
  @IsString()
  cohortId?: string | ObjectId;

  @JSONSchema({
    description: 'Number of faces detected by the model at capture time',
    type: 'integer',
    example: 1,
  })
  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(0)
  faceCount?: number;

  @JSONSchema({
    description:
      'Per-face detection confidence (0-1) reported by the face detection model, one entry per detected face. Sent as a JSON-encoded array string in multipart requests.',
    type: 'array',
    items: {type: 'number'},
  })
  @IsOptional()
  @Transform(({value}) => {
    if (typeof value !== 'string') return value;
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  })
  @IsArray()
  @IsNumber({}, {each: true})
  @Min(0, {each: true})
  @Max(1, {each: true})
  confidenceScores?: number[];
}

export class AnomalyData extends NewAnomalyData implements IAnomalyData {
  @JSONSchema({
    description: 'Unique identifier for the anomaly',
    type: 'string',
    readOnly: true,
  })
  @IsOptional()
  @IsMongoId()
  _id?: string | ObjectId;

  @JSONSchema({
    description: 'User ID associated with the anomaly',
    type: 'string',
    readOnly: true,
  })
  @IsMongoId()
  @IsString()
  userId: string | ObjectId;

  @JSONSchema({
    description: 'Full name of the student who triggered the anomaly',
    type: 'string',
    readOnly: true,
    example: 'John Doe',
  })
  @IsOptional()
  @IsString()
  studentName?: string;

  @JSONSchema({
    description: 'Email of the student who triggered the anomaly',
    type: 'string',
    readOnly: true,
    example: 'john.doe@example.com',
  })
  @IsOptional()
  @IsEmail()
  studentEmail?: string;

  @JSONSchema({
    description: 'URL of the anomaly image stored in cloud storage',
    type: 'string',
    readOnly: true,
  })
  @IsOptional()
  @IsString()
  fileName?: string;

  @JSONSchema({
    description: 'Type of the file associated with the anomaly',
    type: 'string',
    enum: Object.values(FileType),
    readOnly: true,
  })
  @IsEnum(FileType)
  @IsOptional()
  @IsString()
  fileType?: FileType;

  @JSONSchema({
    description: 'Timestamp when the anomaly was detected',
    type: 'string',
    format: 'date-time',
    readOnly: true,
  })
  @IsNotEmpty()
  createdAt: Date;

  @JSONSchema({
    description: 'Cohort name associated with the anomaly (optional)',
    type: 'string',
    readOnly: true,
  })
  @IsOptional()
  @IsString()
  cohortName?: string;
}

export class GetCourseAnomalyParams {
  @JSONSchema({
    description: 'Course ID to filter anomalies',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  courseId: string;

  @JSONSchema({
    description: 'Version ID to filter anomalies',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  versionId: string;
}

export class GetUserAnomalyParams extends GetCourseAnomalyParams {
  @JSONSchema({
    description: 'User ID to filter anomalies',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  userId: string;
}

export class GetItemAnomalyParams extends GetCourseAnomalyParams {
  @JSONSchema({
    description: 'Item ID to filter anomalies',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  itemId: string;
}

export class AnomalyIdParams {
  @JSONSchema({
    description: 'Anomaly ID to identify the anomaly record',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  id: string;
}

export class DeleteAnomalyBody {
  @JSONSchema({
    description: 'Course ID associated with the anomaly to be deleted',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  courseId: string;

  @JSONSchema({
    description: 'Version ID associated with the anomaly to be deleted',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  versionId: string;
}

export class GetAnomalyParams extends GetCourseAnomalyParams {
  @JSONSchema({
    description: 'Anomaly ID to retrieve specific anomaly details',
    type: 'string',
  })
  @IsNotEmpty()
  @IsMongoId()
  @IsString()
  anomalyId: string;
}

export class StatsQueryParams {
  @JSONSchema({
    description: 'Item ID to filter anomaly statistics',
    type: 'string',
  })
  @IsOptional()
  @IsMongoId()
  @IsString()
  itemId?: string;

  @JSONSchema({
    description: 'User ID to filter anomaly statistics',
    type: 'string',
  })
  @IsOptional()
  @IsMongoId()
  @IsString()
  userId?: string;
}

export class CourseAnomaliesQuery extends PaginationWithSortQuery {
  @JSONSchema({
    description: 'Search term to filter anomalies by student name or email',
    type: 'string',
    example: 'john',
  })
  @IsOptional()
  @IsString()
  search?: string;

  @JSONSchema({
    description: 'Filter anomalies by type',
    enum: Object.values(AnomalyType),
    example: AnomalyType.VOICE_DETECTION,
  })
  @IsOptional()
  @IsEnum(AnomalyType)
  type?: AnomalyType;

  @JSONSchema({
    description: 'Filter anomalies by cohort ID',
    type: 'string',
  })
  @IsOptional()
  @IsMongoId()
  @IsString()
  cohort?: string;
}

export {SortOrder, PaginationWithSortQuery};
