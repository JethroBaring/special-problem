import { Prisma } from '@prisma/client';

export class CreateCandidateDto {
  firstName: string;
  lastName: string;
  bio?: string | null;
  resumeUrl?: string | null;
  skills?: Prisma.CandidateCreateskillsInput | string[];
  createdAt?: Date | string;
  updatedAt?: Date | string;
  userId: string;
  jobApplications?: Prisma.JobApplicationCreateNestedManyWithoutApplicantInput;
}
