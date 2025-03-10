import { Prisma } from '@prisma/client';

export class CreateUserDto {
  id?: string;
  email: string;
  password: string;
  refreshToken?: string | null;
  role: UserRole;
  createdAt?: Date | string;
  updatedAt?: Date | string;
  deletedAt?: Date | string | null;
  candidate?: Prisma.CandidateCreateNestedOneWithoutUserInput;
  company?: Prisma.CompanyCreateNestedOneWithoutUserInput;
  notifications?: Prisma.NotificationCreateNestedManyWithoutRecipientInput;
}

export enum UserRole {
  CANDIDATE = 'CANDIDATE',
  COMPANY = 'COMPANY',
}
