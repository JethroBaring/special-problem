import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { UsersModule } from './users/users.module';
import { PrismaModule } from 'nestjs-prisma';
import { NotificationsModule } from './notifications/notifications.module';
import { JobsModule } from './jobs/jobs.module';
import { JobPostingsModule } from './job-postings/job-postings.module';
import { JobApplicationsModule } from './job-applications/job-applications.module';
import { CompanyModule } from './companies/company.module';
import { CandidateModule } from './candidates/candidate.module';
import { AuthModule } from './auth/auth.module';

@Module({
  imports: [
    PrismaModule.forRootAsync({
      isGlobal: true,
      useFactory: () => ({
        prismaOptions: {
          log: ['info', 'query'],
        },
        explicitConnect: false,
      }),
    }),
    UsersModule,
    NotificationsModule,
    JobsModule,
    JobPostingsModule,
    JobApplicationsModule,
    CompanyModule,
    CandidateModule,
    AuthModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
