import { Prisma } from '@prisma/client';

export class CreateJobApplicationDto {
  status?: ApplicationStatus;
  appliedAt?: Date | string;
  createdAt?: Date | string;
  updatedAt?: Date | string;
  deletedAt?: Date | string | null;
  job: Prisma.JobPostingCreateNestedOneWithoutApplicationsInput;
  applicant: Prisma.CandidateCreateNestedOneWithoutJobApplicationsInput;
}

export enum ApplicationStatus {
  PENDING = 'PENDING',
  ACCEPTED = 'ACCEPTED',
  REJECTED = 'REJECTED',
}
