import { Prisma } from '@prisma/client';

export class CreateJobPostingDto {
  id?: string;
  title: string;
  description: string;
  requirements?: string | null;
  location?: string | null;
  salary?: string | null;
  jobType: JobType;
  status?: JobStatus;
  JobCategory?: Prisma.JobCategoryCreateNestedManyWithoutJobPostingsInput;
}

export enum JobType {
  FULL_TIME = 'FULL_TIME',
  PART_TIME = 'PART_TIME',
  CONTRACT = 'CONTRACT',
  TEMPORARY = 'TEMPORARY',
  INTERNSHIP = 'INTERNSHIP',
}

export enum JobStatus {
  OPEN = 'OPEN',
  CLOSED = 'CLOSED',
}
